
"""
Implemented negative log likelihood loss function for ResNet50-CoxRegression model

"""
import tensorflow as tf
from typing import Any, Dict, Iterable, Tuple, Optional, Union

def negative_loglike_loss(cox_output =None, censoring =None,riskset=None):
    """
    Two ways to compute negative log likelihood loss:
        a) Sorting the surival time in descending order
        b) Computing riskset without changing the order of data : Currently used  

    params cox_output : Predicted linear cox output from resnet model
    params censoring : censored(event) vector 
    params riskset : computed risk set from survival time

    """
    with tf.variable_scope('cox_neglogloss'):
        censoring = tf.cast(censoring, cox_output.dtype)
        normalized_pred = safe_normalize(cox_output)
        normalized_pred_t = tf.transpose(normalized_pred)
        #compute log of sum over risk set for each row
        log_risk = logsumexp_masked(normalized_pred_t,riskset,axis=1,keepdims=True)
        censored_loss = tf.multiply(censoring,log_risk-normalized_pred)
        num_observed_events = tf.reduce_sum(censoring)
        loss = tf.reduce_sum(censored_loss)/num_observed_events
    return loss ,normalized_pred



def safe_normalize(x: tf.Tensor) -> tf.Tensor:
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    
    return x + norm

def logsumexp_masked(risk_scores: tf.Tensor,
                     mask: tf.Tensor,
                     axis: int = 0,
                     keepdims: Optional[bool] = None) -> tf.Tensor:
    """Compute logsumexp across `axis` for entries where `mask` is true."""
#     risk_scores.get_shape().assert_same_rank(mask.get_shape())

    with tf.name_scope("logsumexp_masked", values=[risk_scores, mask]):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.multiply(risk_scores, mask_f)
        # for numerical stability, substract the maximum value
        # before taking the exponential

        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax

        exp_masked = tf.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        output = amax + tf.log(exp_sum)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
    return output
