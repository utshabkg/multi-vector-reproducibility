**CRITICAL ISSUE IDENTIFIED**

The experiment has been stuck for 15+ hours on retrieval because:

## Problem
- Brute-force MaxSim on 8.8M documents is computationally infeasible
- Small-scale test (10K docs): 167ms/query
- Full scale (8.8M docs = 880x larger): **~2.5 minutes/query**
- Total time for 6,980 queries: **~12 DAYS**

## Current Status
- Process running, using 291% CPU, 262GB RAM
- Stuck on retrieval phase (likely first few queries)
- No progress output for 15+ hours

## Solution Needed
The paper uses PLAID indexing for efficient retrieval. We need to implement:

1. **Option A**: Use FAISS for approximate nearest neighbor search
2. **Option B**: Optimize with chunked processing + progress tracking
3. **Option C**: Contact authors for implementation details

## Immediate Actions
1. Kill current process
2. Implement FAISS-based retrieval
3. Test on small subset first
4. Re-run with efficient indexing

## Note
The index size is also wrong: 67GB instead of expected 11GB. This suggests our storage format is inefficient (pickle includes overhead).
