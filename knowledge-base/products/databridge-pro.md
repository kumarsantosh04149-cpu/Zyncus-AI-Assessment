# DataBridge Pro

## Connectors Module

DataBridge Pro connects to external data sources via the Connectors module.

---

## Error: ERR_CONNECTION_TIMEOUT

This error occurs when a connector cannot reach the external data source within
30 seconds. Common causes: firewall rules blocking outbound traffic, expired
credentials, or source system downtime.

**Resolution steps:**
1. Verify network connectivity to the source system.
2. Re-authenticate the connector credentials.
3. Check source system status page.

---

## Bulk Operations in Data Ingestion

As of v3.0, Data Ingestion supports bulk archive of entries via multi-select
in the UI, available on Professional plan and above. Starter plan is limited
to single-item operations.
