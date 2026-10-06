"""Conservative delimiter recovery after the existing format-repair request."""
import json

from cd_llm.experiments import neuropathic_new_prompt_cache as prompts
from cd_llm.experiments.alarm_claim_pools_local_transport import request_identity
from cd_llm.baselines.causal_order2025.openrouter_backend import _request_key

POLICY = 'append-one-missing-final-object-brace-after-permitted-repair-v1'


def recover_final_brace(record):
    """Accept only a complete, flat answer object missing its final delimiter."""
    choices = record.get('response', {}).get('choices') or []
    if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
        return None
    content = record['content'].strip()
    if not content.startswith('{') or content.endswith('}'):
        return None
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result
    try:
        value = json.loads(content+'}', object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        return None
    if (not isinstance(value, dict) or set(value) != {'answer','confidence','justification'}
            or not isinstance(value['answer'], str) or not isinstance(value['justification'], str)
            or isinstance(value['confidence'], (dict,list))):
        return None
    parsed = prompts.local.parse_record(dict(record,content=content+'}'),'stated')
    if parsed['status'] not in ('valid','abstain'):
        return None
    return dict(parsed, syntactic_recovery=POLICY)


async def elicit(query, strategy, chat, inputs):
    # Strict parsing still determines whether the one paid/generated repair is
    # required. Recover only that final response, so all its costs are retained.
    row = await prompts.elicit(query,strategy,chat,inputs)
    if strategy!='stated' or len(row['decisions'])!=2 or row['decisions'][-1]['status']!='malformed':
        return row
    messages = prompts.messages(query,strategy,inputs,retry=True)
    request = request_identity(messages,chat.manifest)
    if hasattr(chat,'cache'):
        record = chat.cache.get(request)
    else:
        record = chat.records[_request_key(request)]['record']
    if record is None or record['request_key']!=row['request_keys'][-1]:
        raise ValueError('repair response is missing from the cache')
    parsed = recover_final_brace(record)
    if parsed is None:
        return row
    row['decisions'][-1] = parsed
    row.update(answer=parsed['answer'],reported_confidence=parsed['reported_confidence'],
               valid=parsed['answer'] is not None,
               abstention_reason=None if parsed['answer'] is not None else 'abstain')
    return row
