import tensorflow as tf

class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(input_tensor=tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(input_tensor=xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.compat.v1.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        # training op
        train_op = opt.minimize(loss+lossL2)
        
        return train_op

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(input=logits, axis=1)
        return tf.math.confusion_matrix(labels=labels, predictions=preds)

##########################
# Adapted from tkipf/gcn #
##########################

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(labels))
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(input_tensor=mask)
        loss *= mask
        return tf.reduce_mean(input_tensor=loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss=tf.reduce_mean(input_tensor=loss,axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(input_tensor=mask)
        loss *= mask
        return tf.reduce_mean(input_tensor=loss)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(input_tensor=mask)
        accuracy_all *= mask
        return tf.reduce_mean(input_tensor=accuracy_all)

    def micro_f1(logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)
        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.math.count_nonzero(predicted * labels * mask)
        tn = tf.math.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.math.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.math.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure
