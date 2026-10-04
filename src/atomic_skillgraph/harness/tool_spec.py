"""One public definition per tool; handlers remain owned by their adapter."""
from dataclasses import dataclass, field


def result_schema(data=None):
    return {'type': 'object', 'properties': {
        'accepted': {'type': 'boolean'}, 'observation': {}, 'data': data or {},
        'error': {}, 'done': {'type': 'boolean'}},
        'required': ['accepted', 'observation', 'data', 'error', 'done']}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    result_schema: dict
    effect: str = 'stateful'
    batchable: bool = False
    field_units: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.effect not in {'read_only', 'stateful', 'sandbox_compute'}:
            raise ValueError('Unknown tool effect')
        if self.batchable and self.effect != 'read_only':
            raise ValueError('Only read-only tools can be batched')

    def view(self, **dynamic):
        return {'name': self.name, 'description': self.description,
                'input_schema': self.input_schema, 'result_schema': self.result_schema,
                'effect': self.effect, 'batchable': self.batchable,
                'field_units': self.field_units, **dynamic}
