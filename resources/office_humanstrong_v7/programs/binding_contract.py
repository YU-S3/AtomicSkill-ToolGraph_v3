def validate_bindings(tables, bindings):
    """Reject key mismatches before spending any source-read calls."""
    keys = [table['key'] for table in tables]
    if len(set(keys)) != len(keys):
        raise SelectionError('duplicate_table_key', 'tables[].key must be unique.', field='tables.key')
    if not isinstance(bindings, dict) or not bindings or any(
            not isinstance(names, list) or not names or
            any(not isinstance(key, str) for key in names) or len(set(names)) != len(names)
            for names in bindings.values()):
        raise SelectionError('invalid_bindings', 'Each operand maps to a nonempty ordered list of distinct table keys.',
                             field='bindings', allowed_table_keys=keys)
    bound = {key for names in bindings.values() for key in names}
    if bound != set(keys):
        raise SelectionError('unknown_or_unused_table_key',
            'bindings references tables[].key, not row labels, column names or values.',
            field='bindings', allowed_table_keys=keys,
            unknown_keys=sorted(bound-set(keys)), unused_keys=sorted(set(keys)-bound))
    return bindings
