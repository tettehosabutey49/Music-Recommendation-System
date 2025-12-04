"""
DESIGN DECISION: Redis Configuration
=====================================

Deployment: Redis Cluster (3 masters, 3 replicas)
Why not single instance?
- Single point of failure
- Limited memory (max 64GB per instance)
- Can't handle >100K QPS

Why Redis over alternatives?
1. Memcached: No persistence, simpler data structures
2. DynamoDB: 10x more expensive, higher latency
3. In-memory only: Doesn't scale across instances

Configuration:
- Eviction: allkeys-lru (auto-remove least used)
- Persistence: RDB snapshots every 5 min + AOF
- Memory: 16GB per instance (50K users × 100 recs × 200 bytes)
- TTL: 1 hour for recommendations (balance freshness vs hit rate)

Cost Analysis:
- Redis Cloud: $100/month for 16GB cluster
- DynamoDB: $500/month for same throughput
- Managed vs self-hosted: +50% cost, -90% ops burden
"""