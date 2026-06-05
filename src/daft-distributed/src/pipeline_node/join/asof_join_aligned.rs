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
use futures::TryStreamExt;
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
        scheduler::SchedulerHandle,
        task::{SwordfishTask, SwordfishTaskBuilder},
    },
    statistics::stats::RuntimeStatsRef,
    utils::channel::{Sender, create_channel},
};

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
        right_partitions: Vec<MaterializedOutput>,
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

        let right_psets = right_partitions
            .into_iter()
            .flat_map(|output| output.into_inner().0)
            .collect::<Vec<_>>();

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
        result_tx: &Sender<SwordfishTaskBuilder>,
    ) -> DaftResult<()> {
        let num_partitions = left_partitioned_outputs.len();

        for (i, left_partition) in left_partitioned_outputs.into_iter().enumerate() {
            let right_partitions: Vec<MaterializedOutput> = match self.strategy {
                AsofJoinStrategy::Backward => {
                    let mut v = vec![];
                    if i > 0 {
                        v.push(right_partitioned_outputs[i - 1].clone());
                    }
                    v.push(right_partitioned_outputs[i].clone());
                    v
                }
                AsofJoinStrategy::Forward => {
                    let mut v = vec![right_partitioned_outputs[i].clone()];
                    if i < num_partitions - 1 {
                        v.push(right_partitioned_outputs[i + 1].clone());
                    }
                    v
                }
                AsofJoinStrategy::Nearest => {
                    let mut v = vec![];
                    if i > 0 {
                        v.push(right_partitioned_outputs[i - 1].clone());
                    }
                    v.push(right_partitioned_outputs[i].clone());
                    if i < num_partitions - 1 {
                        v.push(right_partitioned_outputs[i + 1].clone());
                    }
                    v
                }
            };
            self.create_and_submit_join_task(left_partition, right_partitions, result_tx)
                .await?;
        }

        Ok(())
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

        self.zip_and_join(left_materialized, right_materialized, &result_tx)
            .await
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
