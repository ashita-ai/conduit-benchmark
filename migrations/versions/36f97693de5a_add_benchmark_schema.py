"""Add benchmark schema

Revision ID: 36f97693de5a
Revises: 
Create Date: 2025-11-25 19:46:40.401829

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '36f97693de5a'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create benchmark_runs table
    op.create_table(
        'benchmark_runs',
        sa.Column('benchmark_id', sa.String(length=36), nullable=False),
        sa.Column('dataset_size', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('benchmark_id')
    )
    op.create_index(op.f('ix_benchmark_runs_created_at'), 'benchmark_runs', ['created_at'], unique=False)

    # Create algorithm_runs table
    op.create_table(
        'algorithm_runs',
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('benchmark_id', sa.String(length=36), nullable=False),
        sa.Column('algorithm_name', sa.String(length=50), nullable=False),
        sa.Column('total_cost', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('average_quality', sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column('total_queries', sa.Integer(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['benchmark_id'], ['benchmark_runs.benchmark_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('run_id')
    )
    op.create_index(op.f('ix_algorithm_runs_benchmark_id'), 'algorithm_runs', ['benchmark_id'], unique=False)
    op.create_index(op.f('ix_algorithm_runs_algorithm_name'), 'algorithm_runs', ['algorithm_name'], unique=False)

    # Create query_evaluations table
    op.create_table(
        'query_evaluations',
        sa.Column('evaluation_id', sa.String(length=36), nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('query_id', sa.String(length=36), nullable=False),
        sa.Column('model_id', sa.String(length=100), nullable=False),
        sa.Column('quality_score', sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column('cost', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('latency', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['algorithm_runs.run_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('evaluation_id')
    )
    op.create_index(op.f('ix_query_evaluations_run_id'), 'query_evaluations', ['run_id'], unique=False)
    op.create_index(op.f('ix_query_evaluations_query_id'), 'query_evaluations', ['query_id'], unique=False)
    op.create_index(op.f('ix_query_evaluations_model_id'), 'query_evaluations', ['model_id'], unique=False)
    op.create_index(op.f('ix_query_evaluations_created_at'), 'query_evaluations', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_query_evaluations_created_at'), table_name='query_evaluations')
    op.drop_index(op.f('ix_query_evaluations_model_id'), table_name='query_evaluations')
    op.drop_index(op.f('ix_query_evaluations_query_id'), table_name='query_evaluations')
    op.drop_index(op.f('ix_query_evaluations_run_id'), table_name='query_evaluations')
    op.drop_table('query_evaluations')

    op.drop_index(op.f('ix_algorithm_runs_algorithm_name'), table_name='algorithm_runs')
    op.drop_index(op.f('ix_algorithm_runs_benchmark_id'), table_name='algorithm_runs')
    op.drop_table('algorithm_runs')

    op.drop_index(op.f('ix_benchmark_runs_created_at'), table_name='benchmark_runs')
    op.drop_table('benchmark_runs')
