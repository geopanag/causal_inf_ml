from sklift.models import SoloModel
from sklift.models import TwoModels
from sklift.models import ClassTransformation

from causalml.inference.meta import BaseXClassifier, BaseTClassifier, BaseSClassifier#, BaseXRegressor, BaseTRegressor, BaseSRegressor
#from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from econml.metalearners import TLearner, SLearner, XLearner
from xgboost import XGBClassifier, XGBRegressor

from sklift.metrics import uplift_at_k, uplift_auc_score, qini_auc_score
import pandas as pd



def sklift_dml(X, y, X_t,y_t, label_l, model=XGBClassifier):

    if label_l =='S':

        sm = SoloModel(model()) 
        mod = sm.fit(X.drop('treatment',axis=1), y , X['treatment'])#, estimator_fit_params={'cat_features': cat_features}
        
    elif label_l =='T':
        tm = TwoModels(
            estimator_trmnt=model(), 
            estimator_ctrl=model(), 
            method='vanilla'
        )
        mod = tm.fit(X.drop('treatment',axis=1), y , X['treatment'])
    else: 
        tm = TwoModels(
            estimator_trmnt=model(), 
            estimator_ctrl=model(), 
            method='ddr_control'
        )
        mod = tm.fit(X.drop('treatment',axis=1), y , X['treatment'])

    uplift = mod.predict(X_t.drop('treatment',axis=1))
    
    m_score = uplift_at_k(y_true=y_t, uplift=uplift, treatment=X_t['treatment'], strategy='by_group', k=0.4)
    return m_score


def causalml_dml(X,y, X_t,y_t, base_learner,label_l, model_out=XGBClassifier, model_effect=XGBRegressor):
    if label_l == 'S':
        learner = base_learner(learner=model_out())
        ate, lb, ub = learner.estimate_ate( X=X.drop('treatment',axis=1),y=y , treatment=X['treatment'],return_ci=True )  
        uplift=learner.predict(X=X_t.drop('treatment',axis=1), treatment=X_t['treatment']).squeeze()
        
    elif label_l == 'T':
        learner = base_learner(learner=model_out())
        ate, lb, ub = learner.estimate_ate(X=X.drop('treatment',axis=1),y=y , treatment=X['treatment'])  
        
        uplift=learner.predict(X=X_t.drop('treatment',axis=1), treatment=X_t['treatment']).squeeze()
    elif label_l == 'X':
        propensity_model = XGBClassifier()

        propensity_model.fit(X=X.drop('treatment',axis=1), y=X['treatment'])
        p_train = propensity_model.predict_proba(X=X.drop('treatment',axis=1))
        p_test = propensity_model.predict_proba(X=X_t.drop('treatment',axis=1))

        p_train = pd.Series(p_train[:, 0])
        p_test  = pd.Series(p_test[:, 0])        
        learner = base_learner(
        outcome_learner=model_out(),
        effect_learner=model_effect())
            
        ate, lb, ub = learner.estimate_ate(X=X.drop('treatment',axis=1) ,y=y, treatment=X['treatment'], p=p_train)  
        uplift=learner.predict(X=X_t.drop('treatment',axis=1) , treatment=X_t['treatment'], p=p_test).squeeze()
        #preds_dict['{} Learner ({})'.format(label_l, label_m)] = np.ravel([ate, lb, ub])

    score = uplift_at_k(y_true=y_t, uplift=uplift, treatment=X_t['treatment'], strategy='by_group', k=0.4)
    #auc_score = uplift_auc_score(y_true=y_t, uplift=y_pred, treatment=X_t['treatment'])
    #qini_score = qini_auc_score(y_true=y_t, uplift=y_pred, treatment=X_t['treatment'])
    return score#, auc_score, qini_score



def econml_dml(X,y, X_t,y_t,label_l, model_out=XGBClassifier, model_effect=XGBRegressor):

    if label_l == 'S':
        sl = SLearner(overall_model=model_out())
        sl.fit(y, X['treatment'], X=X.drop(columns='treatment',axis=1))
        uplift = sl.effect(X_t.drop(columns='treatment',axis=1))
    elif label_l == 'T':
        tl = TLearner(models=model_out())
        # Train T_learner
        tl.fit(y, X['treatment'], X=X.drop(columns='treatment',axis=1))
        uplift = tl.effect(X_t.drop(columns='treatment',axis=1))
    else:        
    
        xl = XLearner(models=model_out(),
            propensity_model=model_out(),
            cate_models=model_effect())
            
        xl.fit(y, X['treatment'], X=X.drop(columns='treatment',axis=1))
        # Estimate treatment effects on test data
        uplift = xl.effect(X_t.drop(columns='treatment',axis=1))

    score = uplift_at_k(y_true=y_t, uplift=uplift, treatment=X_t['treatment'], strategy='by_group', k=0.4)
    return score#, auc_score, qini_score