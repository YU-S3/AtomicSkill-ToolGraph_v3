"""Plain public action documentation."""
def tool_summary(adapter):
    return [{'name': tool['name'], 'description': tool.get('description', ''),
             'input_schema': tool['input_schema']} for tool in adapter.available_tools()]
