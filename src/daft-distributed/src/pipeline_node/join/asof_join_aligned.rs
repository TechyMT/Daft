use std::sync::Arc;

use common_error::DaftResult;
use common_metrics::{
    Meter,
    ops::{NodeCategory, NodeType},
};
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_local_plan::{LocalNodeContext, LocalPhysicalPlan, ShuffleReadBackend};
use daft_logical_plan::{AsofJoinStrategy, stats::StatsState};
use daft_schema::schema::SchemaRef;
use futures::{TryStreamExt, future::try_join_all};
use itertools::Itertools;

use super::stats::BasicJoinStats;
use crate::{
    pipeline_node::{
        ClusteringStrategy, DistributedPipelineNode, MaterializedOutput, NodeID,
        PipelineNodeConfig, PipelineNodeContext, PipelineNodeImpl, TaskBuilderStream,
        clustering::BoundClusteringSpec,
    },
    plan::{PlanConfig, PlanExecutionContext, TaskIDCounter},
    scheduling::{
        scheduler::{SchedulerHandle, SubmittedTask},
        task::{SchedulingStrategy, SwordfishTask, SwordfishTaskBuilder},
    },
    statistics::stats::RuntimeStatsRef,
    utils::channel::{Sender, create_channel},
};

const FINAL_CARRYOVER_BACKWARD_PHASE: &str = "final_carryover_backward";
const FINAL_CARRYOVER_FORWARD_PHASE: &str = "final_carryover_forward";

pub(crate) struct AsofJoinAlignedNode {
    config: PipelineNodeConfig,
    context: PipelineNodeContext,

    left_by: Vec<BoundExpr>,
    right_by: Vec<BoundExpr>,
    left_on: BoundExpr,
    right_on: BoundExpr,
    strategy: AsofJoinStrategy,

    left: DistributedPipelineNode,
    right: DistributedPipelineNode,
}

impl AsofJoinAlignedNode {
    const NODE_NAME: &'static str = "AsofJoinAligned";

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_id: NodeID,
        plan_config: &PlanConfig,
        left_by: Vec<BoundExpr>,
        right_by: Vec<BoundExpr>,
        left_on: BoundExpr,
        right_on: BoundExpr,
        strategy: AsofJoinStrategy,
        left: DistributedPipelineNode,
        right: DistributedPipelineNode,
        output_schema: SchemaRef,
    ) -> Self {
        let context = PipelineNodeContext::new(
            plan_config.query_idx,
            plan_config.query_id.clone(),
            node_id,
            Arc::from(Self::NODE_NAME),
            NodeType::AsofJoin,
            NodeCategory::BlockingSink,
        );
        let num_partitions = left.config().clustering_spec.num_partitions();
        let config = PipelineNodeConfig::new(
            output_schema,
            plan_config.config.clone(),
            ClusteringStrategy::Explicit(BoundClusteringSpec::unknown(num_partitions)),
        );
        Self {
            config,
            context,
            left_by,
            right_by,
            left_on,
            right_on,
            strategy,
            left,
            right,
        }
    }

    async fn create_and_submit_join_task(
        self: &Arc<Self>,
        left_partition: MaterializedOutput,
        right_partition: MaterializedOutput,
        carryovers: (Option<MaterializedOutput>, Option<MaterializedOutput>),
        result_tx: &Sender<SwordfishTaskBuilder>,
    ) -> DaftResult<()> {
        let left_shuffle_read_plan = LocalPhysicalPlan::shuffle_read(
            self.left.node_id(),
            self.left.config().schema.clone(),
            ShuffleReadBackend::Ray,
            StatsState::NotMaterialized,
            LocalNodeContext::new(Some(self.left.node_id() as usize)),
        );

        let left_psets = left_partition.into_inner().0;

        let right_shuffle_read_plan = LocalPhysicalPlan::shuffle_read(
            self.right.node_id(),
            self.right.config().schema.clone(),
            ShuffleReadBackend::Ray,
            StatsState::NotMaterialized,
            LocalNodeContext::new(Some(self.right.node_id() as usize)),
        );

        // Assemble right psets in ascending ts order: [backward | partition | forward]
        let (backward_carryover, forward_carryover) = carryovers;
        let mut right_psets = Vec::new();
        if let Some(carryover) = backward_carryover {
            right_psets.extend(carryover.into_inner().0);
        }
        right_psets.extend(right_partition.into_inner().0);
        if let Some(carryover) = forward_carryover {
            right_psets.extend(carryover.into_inner().0);
        }

        let plan = LocalPhysicalPlan::asof_join(
            left_shuffle_read_plan,
            right_shuffle_read_plan,
            self.left_by.clone(),
            self.right_by.clone(),
            self.left_on.clone(),
            self.right_on.clone(),
            self.strategy,
            self.config.schema.clone(),
            StatsState::NotMaterialized,
            LocalNodeContext::new(Some(self.node_id() as usize)),
        );

        let builder = SwordfishTaskBuilder::new(plan, self.as_ref(), self.node_id())
            .with_psets(self.left.node_id(), left_psets)
            .with_psets(self.right.node_id(), right_psets);

        result_tx.send(builder).await.ok();
        Ok(())
    }

    async fn zip_and_join(
        self: Arc<Self>,
        left_partitioned_outputs: Vec<MaterializedOutput>,
        right_partitioned_outputs: Vec<MaterializedOutput>,
        task_id_counter: &TaskIDCounter,
        result_tx: &Sender<SwordfishTaskBuilder>,
        scheduler_handle: &SchedulerHandle<SwordfishTask>,
    ) -> DaftResult<()> {
        let num_partitions = left_partitioned_outputs.len();

        if num_partitions == 1 {
            let left = left_partitioned_outputs.into_iter().next().unwrap();
            let right = right_partitioned_outputs.into_iter().next().unwrap();
            return self
                .create_and_submit_join_task(left, right, (None, None), result_tx)
                .await;
        }

        let (backward_carryovers, forward_carryovers) = match self.strategy {
            AsofJoinStrategy::Backward => {
                let backward_carryovers = self
                    .compute_carryovers_aligned(
                        right_partitioned_outputs.clone(),
                        true,
                        task_id_counter,
                        scheduler_handle,
                    )
                    .await?;
                (
                    backward_carryovers,
                    vec![None::<MaterializedOutput>; num_partitions],
                )
            }
            AsofJoinStrategy::Forward => {
                let forward_carryovers = self
                    .compute_carryovers_aligned(
                        right_partitioned_outputs.clone(),
                        false,
                        task_id_counter,
                        scheduler_handle,
                    )
                    .await?;
                (
                    vec![None::<MaterializedOutput>; num_partitions],
                    forward_carryovers,
                )
            }
            AsofJoinStrategy::Nearest => tokio::try_join!(
                self.compute_carryovers_aligned(
                    right_partitioned_outputs.clone(),
                    true,
                    task_id_counter,
                    scheduler_handle,
                ),
                self.compute_carryovers_aligned(
                    right_partitioned_outputs.clone(),
                    false,
                    task_id_counter,
                    scheduler_handle,
                ),
            )?,
        };

        for (i, (left_partition, right_partition)) in left_partitioned_outputs
            .into_iter()
            .zip(right_partitioned_outputs)
            .enumerate()
        {
            let carryovers = (
                if i == 0 {
                    None
                } else {
                    backward_carryovers[i - 1].clone()
                },
                if i == num_partitions - 1 {
                    None
                } else {
                    forward_carryovers[i + 1].clone()
                },
            );
            self.create_and_submit_join_task(
                left_partition,
                right_partition,
                carryovers,
                result_tx,
            )
            .await?;
        }

        Ok(())
    }

    async fn compute_carryovers_aligned(
        &self,
        right_partitioned_outputs: Vec<MaterializedOutput>,
        is_strategy_backward: bool,
        task_id_counter: &TaskIDCounter,
        scheduler_handle: &SchedulerHandle<SwordfishTask>,
    ) -> DaftResult<Vec<Option<MaterializedOutput>>> {
        let descending = is_strategy_backward;
        let propagate_forward = is_strategy_backward;

        let final_carryover_tasks = self.create_final_carryover_tasks(
            right_partitioned_outputs,
            descending,
            task_id_counter,
            scheduler_handle,
        )?;

        let mut final_carryovers: Vec<Option<MaterializedOutput>> =
            try_join_all(final_carryover_tasks.into_iter().map(|t| async {
                match t {
                    Some(task) => task.await.map(|mo| mo.filter(|m| m.num_rows() > 0)),
                    None => Ok(None),
                }
            }))
            .await?;

        let n = final_carryovers.len();
        if propagate_forward {
            for i in 1..n {
                if final_carryovers[i].is_none() {
                    let prev = final_carryovers[i - 1].clone();
                    final_carryovers[i] = prev;
                }
            }
        } else {
            for i in (0..n.saturating_sub(1)).rev() {
                if final_carryovers[i].is_none() {
                    let next = final_carryovers[i + 1].clone();
                    final_carryovers[i] = next;
                }
            }
        }

        Ok(final_carryovers)
    }

    async fn execution_loop(
        self: Arc<Self>,
        left_inputs: TaskBuilderStream,
        right_inputs: TaskBuilderStream,
        task_id_counter: TaskIDCounter,
        result_tx: Sender<SwordfishTaskBuilder>,
        scheduler_handle: SchedulerHandle<SwordfishTask>,
    ) -> DaftResult<()> {
        let left_materialized = left_inputs
            .materialize(
                scheduler_handle.clone(),
                self.context.query_idx,
                task_id_counter.clone(),
            )
            .try_collect::<Vec<_>>()
            .await?;

        let right_materialized = right_inputs
            .materialize(
                scheduler_handle.clone(),
                self.context.query_idx,
                task_id_counter.clone(),
            )
            .try_collect::<Vec<_>>()
            .await?;

        if left_materialized.len() != right_materialized.len() {
            return Err(common_error::DaftError::InternalError(format!(
                "AsofJoinAligned: partition count mismatch at execution time: \
                 left={}, right={}. The _assume_sorted_and_aligned guarantee was violated.",
                left_materialized.len(),
                right_materialized.len(),
            )));
        }

        if left_materialized.is_empty() {
            return Ok(());
        }

        self.zip_and_join(
            left_materialized,
            right_materialized,
            &task_id_counter,
            &result_tx,
            &scheduler_handle,
        )
        .await
    }

    fn submit_boundary_carryover_task(
        &self,
        boundary_pset: MaterializedOutput,
        offset: Option<u64>,
        partition_idx: usize,
        phase: &str,
        strategy: Option<SchedulingStrategy>,
        task_id_counter: &TaskIDCounter,
        scheduler_handle: &SchedulerHandle<SwordfishTask>,
    ) -> DaftResult<SubmittedTask> {
        let node_id = self.node_id();

        let (in_memory_scan, psets) = MaterializedOutput::into_in_memory_scan_with_psets_and_phase(
            vec![boundary_pset],
            self.right.config().schema.clone(),
            node_id,
            phase,
        );

        let plan = LocalPhysicalPlan::limit(
            in_memory_scan,
            1,
            offset,
            StatsState::NotMaterialized,
            LocalNodeContext::new(Some(node_id as usize)).with_phase(phase),
        );

        SwordfishTaskBuilder::new(plan, self, node_id)
            // Each carryover task has a structurally identical Limit(1) plan. Fold in
            // direction (forward vs backward) and partition index so every task gets
            // its own pipeline instance and streaming early-termination doesn't kill
            // a pipeline that another task is still trying to use.
            .extend_fingerprint(u32::from(offset.is_some()))
            .extend_fingerprint(partition_idx as u32)
            .with_psets(node_id, psets)
            .with_strategy(strategy)
            .build(self.context().query_idx, task_id_counter)
            .submit(scheduler_handle)
    }

    fn create_final_carryover_tasks(
        &self,
        right_partitions: Vec<MaterializedOutput>,
        descending: bool,
        task_id_counter: &TaskIDCounter,
        scheduler_handle: &SchedulerHandle<SwordfishTask>,
    ) -> DaftResult<Vec<Option<SubmittedTask>>> {
        right_partitions
            .into_iter()
            .enumerate()
            .map(|(partition_idx, partition)| {
                if partition.num_rows() == 0 {
                    return Ok(None);
                }

                let phase = if descending {
                    FINAL_CARRYOVER_BACKWARD_PHASE
                } else {
                    FINAL_CARRYOVER_FORWARD_PHASE
                };

                let offset = if descending {
                    Some(partition.num_rows() as u64 - 1)
                } else {
                    None
                };

                self.submit_boundary_carryover_task(
                    partition,
                    offset,
                    partition_idx,
                    phase,
                    None,
                    task_id_counter,
                    scheduler_handle,
                )
                .map(Some)
            })
            .collect()
    }
}

impl PipelineNodeImpl for AsofJoinAlignedNode {
    fn context(&self) -> &PipelineNodeContext {
        &self.context
    }

    fn config(&self) -> &PipelineNodeConfig {
        &self.config
    }

    fn children(&self) -> Vec<DistributedPipelineNode> {
        vec![self.left.clone(), self.right.clone()]
    }

    fn make_runtime_stats(&self, meter: &Meter) -> RuntimeStatsRef {
        Arc::new(BasicJoinStats::new(meter, self.context()))
    }

    fn multiline_display(&self, _verbose: bool) -> Vec<String> {
        let mut res = vec!["AsofJoin (assume_sorted_and_aligned)".to_string()];
        res.push(format!(
            "Left by: [{}]",
            self.left_by.iter().map(|e| e.to_string()).join(", ")
        ));
        res.push(format!(
            "Right by: [{}]",
            self.right_by.iter().map(|e| e.to_string()).join(", ")
        ));
        res.push(format!("Left on: {}", self.left_on));
        res.push(format!("Right on: {}", self.right_on));
        res
    }

    fn produce_tasks(
        self: Arc<Self>,
        plan_context: &mut PlanExecutionContext,
    ) -> TaskBuilderStream {
        let left_input = self.left.clone().produce_tasks(plan_context);
        let right_input = self.right.clone().produce_tasks(plan_context);
        let (result_tx, result_rx) = create_channel(1);
        plan_context.spawn(self.execution_loop(
            left_input,
            right_input,
            plan_context.task_id_counter(),
            result_tx,
            plan_context.scheduler_handle(),
        ));
        TaskBuilderStream::from(result_rx)
    }
}
