#!/usr/bin/env python
# -*- coding: utf-8 -*-


def extract_code(code, s_line, e_line):
    lines = code.splitlines()
    s_idx, e_idx = 0, 0
    for i, line in enumerate(lines):
        if line == s_line:
            s_idx = i
        elif line == e_line:
            e_idx = i
            break
    return '\n'.join(lines[s_idx + 1: e_idx])


def replace_code(code, cur, tgt):
    return code.replace(cur, tgt)


if __name__ == '__main__':
    # Load TinyNdArray
    with open('./tinyndarray/tinyndarray.h', 'r') as ndarray_f:
        ndarray_code = ndarray_f.read()
        decl_code = extract_code(ndarray_code,
                                 '#ifndef TINYNDARRAY_NO_DECLARATION',
                                 '#endif  // TINYNDARRAY_NO_DECLARATION')
        defi_code = extract_code(ndarray_code,
                                 '#ifdef TINYNDARRAY_IMPLEMENTATION',
                                 '#endif  // TINYNDARRAY_IMPLEMENTATION')

    # Load raw TinyDiff
    with open('./tinydiff.h', 'r') as diff_f:
        diff_code = diff_f.read()
        diff_code = replace_code(diff_code,
                                 '\n' +
                                 '// Declaration of NdArray\n' +
                                 '#undef TINYNDARRAY_H_ONCE\n' +
                                 '#define TINYNDARRAY_NO_NAMESPACE\n' +
                                 '#include "./tinyndarray/tinyndarray.h"\n',
                                 decl_code)
        diff_code = replace_code(diff_code,
                                 '\n' +
                                 '// Definitions of NdArray\n' +
                                 '#undef TINYNDARRAY_H_ONCE\n' +
                                 '#define TINYNDARRAY_NO_NAMESPACE\n' +
                                 '#define TINYNDARRAY_NO_DECLARATION\n' +
                                 '#define TINYNDARRAY_IMPLEMENTATION\n' +
                                 '#include "./tinyndarray/tinyndarray.h"\n',
                                 defi_code)

    # Save combined TinyDiff
    with open('../tinydiff.h', 'w') as diff_f:
        diff_f.write(diff_code)
