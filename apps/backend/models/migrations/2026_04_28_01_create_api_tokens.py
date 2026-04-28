"""Manual migration: create api_tokens table."""
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "api_tokens",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("token_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scopes", sa.Text, nullable=False, default=""),
        sa.Column("revoked", sa.Boolean, nullable=False, default=False),
    )

def downgrade():
    op.drop_table("api_tokens")
