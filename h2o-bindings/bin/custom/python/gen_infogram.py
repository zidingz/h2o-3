rest_api_version = 99  # type: int

def update_param(name, param):
    if name == 'model_algorithm_params':
        param['type'] = 'KeyValue'
        param['default_value'] = None
        return param
    elif name == 'infogram_algorithm_params':
        param['type'] = 'KeyValue'
        param['default_value'] = None
        return param
    return None  # param untouched

def class_extensions():
    def admissibleAttributesFrame(self):
        """
        Get the admissible attributes returned by InfoGram as an H2O Frame.
        :param self: 
        :return: H2OFrame
        """
        keyString = self._model_json["output"]["relevance_cmi_key"]
        if not(keyString==None):
            return h2o.get_frame(keyString)
        else:
            return None
        
overrides = dict(
    infogram_algorithm_params=dict(
        getter="""
if self._parms.get("{sname}") != None:
    infogram_algorithm_params_dict =  ast.literal_eval(self._parms.get("{sname}"))
    for k in infogram_algorithm_params_dict:
        if len(infogram_algorithm_params_dict[k]) == 1: #single parameter
            infogram_algorithm_params_dict[k] = infogram_algorithm_params_dict[k][0]
    return infogram_algorithm_params_dict
else:
    return self._parms.get("{sname}")
""",
        setter="""
assert_is_type({pname}, None, {ptype})
if {pname} is not None and {pname} != "":
    for k in {pname}:
        if ("[" and "]") not in str(infogram_algorithm_params[k]):
            infogram_algorithm_params[k] = [infogram_algorithm_params[k]]
    self._parms["{sname}"] = str(json.dumps({pname}))
else:
    self._parms["{sname}"] = None
"""
    ),

    model_algorithm_params=dict(
        getter="""
if self._parms.get("{sname}") != None:
    model_algorithm_params_dict =  ast.literal_eval(self._parms.get("{sname}"))
    for k in model_algorithm_params_dict:
        if len(model_algorithm_params_dict[k]) == 1: #single parameter
            model_algorithm_params_dict[k] = model_algorithm_params_dict[k][0]
    return model_algorithm_params_dict
else:
    return self._parms.get("{sname}")
""",
        setter="""
assert_is_type({pname}, None, {ptype})
if {pname} is not None and {pname} != "":
    for k in {pname}:
        if ("[" and "]") not in str(model_algorithm_params[k]):
            model_algorithm_params[k] = [model_algorithm_params[k]]
    self._parms["{sname}"] = str(json.dumps({pname}))
else:
    self._parms["{sname}"] = None
"""
    ),
)

doc = dict(
    __class__="""
Given a sensitive/unfair predictors list, InfoGram will add all predictors that contains information on the 
 sensitive/unfair predictors list to the sensitive/unfair predictors list.  It will return a set of predictors that
 do not contain information on the sensitive/unfair list and hence user can build a fair model.  If no sensitive/unfair
 predictor list is given, InfoGram will return a list of core predictors that should be used to build a final model.
 InfoGram can significantly cut down the number of predictors needed to build a model and hence will build a simple
 model that is more interpretable, less susceptible to overfitting, runs faster while providing similar accuracy
 as models built using all attributes.
"""
)
