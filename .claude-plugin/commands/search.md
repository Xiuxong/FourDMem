Search your FourDMem memory for relevant context.

Use the `search_memory` MCP tool with the user's query.
After getting results, summarize the most relevant findings and cite the source memories.

```
search_memory(query="{{USER_QUERY}}", limit=10)
```

Then rate the results:
```
submit_feedback(entity_id=<top result id>, score=1.0)
```
