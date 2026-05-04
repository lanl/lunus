#!/usr/bin/env ccp4-python
import gemmi
# IT92Coef has get_coefs - try it
coef = gemmi.IT92Coef.get_coefs(gemmi.Element('C'))
print(type(coef), coef)
print(dir(coef))
