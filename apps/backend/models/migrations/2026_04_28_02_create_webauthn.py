"""Manual migration: create webauthn_credentials table."""
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "webauthn_credentials",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("credential_id", sa.LargeBinary(128), nullable=False, unique=True),
        sa.Column("public_key", sa.LargeBinary, nullable=False),
        sa.Column("sign_count", sa.Integer, default=0),
        sa.Column("transports", sa.String(100), default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_backup", sa.Boolean, default=False),
    )

def downgrade():
    op.drop_table("webauthn_credentials")
