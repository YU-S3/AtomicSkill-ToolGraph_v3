"""Exact determinant of an explicit square rational matrix, dimension at most four."""
from fractions import Fraction
from itertools import permutations


def run(ctx, inputs):
    try:
        rows=inputs['matrix'];n=len(rows)
        if not 1<=n<=4 or any(len(row)!=n for row in rows): raise ValueError('square matrix required')
        def rational(x):
            if not isinstance(x,str) or len(x)>32:raise ValueError('bounded rational string required')
            return Fraction(x)
        a=[[rational(x) for x in row] for row in rows];det=Fraction(0)
        for perm in permutations(range(n)):
            inversions=sum(perm[i]>perm[j] for i in range(n) for j in range(i+1,n))
            term=Fraction((-1)**inversions)
            for i,j in enumerate(perm):term*=a[i][j]
            det+=term
        return {'status':'ok','outputs':{'determinant':str(det),'dimension':n,'scope':'exact_supplied_matrix_only'}}
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        return {'status':'needs_input','outputs':{}}
