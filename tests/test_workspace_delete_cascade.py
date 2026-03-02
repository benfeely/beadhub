"""Tests that workspace soft-delete cascades to aweb agent soft-delete."""

from __future__ import annotations

from uuid import UUID

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from beadhub.api import create_app


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


@pytest.mark.asyncio
async def test_workspace_delete_cascades_to_agent(db_infra, redis_client_async):
    """DELETE /v1/workspaces/{id} should also soft-delete the aweb agent record."""
    app = create_app(db_infra=db_infra, redis=redis_client_async, serve_frontend=False)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Create agent + workspace
            init = await client.post(
                "/v1/init",
                json={
                    "project_slug": "ws-cascade-test",
                    "alias": "ws-cascade-agent",
                    "agent_type": "agent",
                    "repo_origin": "git@github.com:test/ws-cascade.git",
                },
            )
            assert init.status_code == 200, init.text
            data = init.json()
            api_key = data["api_key"]
            agent_id = data["agent_id"]

            # Verify agent is active in aweb
            aweb_db = db_infra.get_manager("aweb")
            agent = await aweb_db.fetch_one(
                """
                SELECT status, deleted_at
                FROM {{tables.agents}}
                WHERE agent_id = $1 AND deleted_at IS NULL
                """,
                UUID(agent_id),
            )
            assert agent is not None, "Agent should exist and be active"

            # Get the workspace_id
            ws_list = await client.get(
                "/v1/workspaces",
                headers=_auth_headers(api_key),
            )
            assert ws_list.status_code == 200, ws_list.text
            workspaces = ws_list.json()["workspaces"]
            assert len(workspaces) == 1
            workspace_id = workspaces[0]["workspace_id"]

            # Delete the workspace
            delete_resp = await client.delete(
                f"/v1/workspaces/{workspace_id}",
                headers=_auth_headers(api_key),
            )
            assert delete_resp.status_code == 200, delete_resp.text

            # Agent should now be soft-deleted
            agent_after = await aweb_db.fetch_one(
                """
                SELECT status, deleted_at
                FROM {{tables.agents}}
                WHERE agent_id = $1 AND deleted_at IS NULL
                """,
                UUID(agent_id),
            )
            assert agent_after is None, "Agent should be soft-deleted after workspace delete"

            # Verify the agent still exists but with deleted_at set
            agent_deleted = await aweb_db.fetch_one(
                """
                SELECT status, deleted_at
                FROM {{tables.agents}}
                WHERE agent_id = $1
                """,
                UUID(agent_id),
            )
            assert agent_deleted is not None, "Agent record should still exist"
            assert agent_deleted["deleted_at"] is not None
            assert agent_deleted["status"] == "deregistered"
