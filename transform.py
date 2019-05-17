def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs
	
def positional_encoding(inputs,
                        num_units,
                        zero_pad = True,
                        scale = True,
                        scope = "positional_encoding",
                        reuse=None):

    N,T = inputs.get_shape().as_list()
    with tf.variable_scope(scope,reuse=True):
        position_ind = tf.tile(tf.expand_dims(tf.range(T),0),[N,1])

        position_enc = np.array([
            [pos / np.power(10000, 2.*i / num_units) for i in range(num_units)]
            for pos in range(T)])

        position_enc[:,0::2] = np.sin(position_enc[:,0::2]) # dim 2i
        position_enc[:,1::2] = np.cos(position_enc[:,1::2]) # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1,num_units]),lookup_table[1:,:]),0)

        outputs = tf.nn.embedding_lookup(lookup_table,position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs


#######################################
# Scaled Dot-Product Attention(self-attention)

def scaled_dotproduct_attention(queries,keys,num_units=None,    #[batch_size,T,embedding_size]
                        num_heads = 0,
                        dropout_rate = 0,
                        is_training = True,
                        causality = False,
                        scope = "mulithead_attention",
                        reuse = None):
    with tf.variable_scope(scope,reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projection
        Q = tf.layers.dense(queries,num_units,activation=tf.nn.relu) # [batch_size,T,num_units]
        K = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #
        V = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #

        outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)    # [batch_size,T,T]

        # 这里是对填充的部分进行一个mask，这些位置的attention score变为极小，我们的embedding操作中是有一个padding操作的，
        # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys,axis=-1)))      #[batch_size,T]
        key_masks = tf.tile(tf.expand_dims(key_masks,1),[1,tf.shape(queries)[1],1])  # [batch_size,T,T]

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks,0),paddings,outputs)   # 填充部分的列会变成0

        # 这里其实就是进行一个mask操作，不给模型看到未来的信息。
        if causality:
            diag_vals = tf.ones_like(outputs[0,:,:])
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(outputs)[0],1,1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks,0),paddings,outputs)

        outputs = tf.nn.softmax(outputs)
        # Query Mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries,axis=-1)))   # [batch_size,T]
        query_masks = tf.tile(tf.expand_dims(query_masks,-1),[1,1,tf.shape(keys)[1]])   # [batch_size,T,T]    # 填充部分的行会变成0
        outputs *= query_masks    # 注意力矩阵的填充部分的行和列都mask为0
        # Dropout
        outputs = tf.layers.dropout(outputs,rate = dropout_rate,training = tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs,V)   # [batch_size,T,num_units]
        # Residual connection
        outputs += queries
        # Normalize
        outputs = normalize(outputs)

    return outputs


#######################################
# multihead_attention

def multihead_attention(queries,keys,num_units=None,
                        num_heads = 0,
                        dropout_rate = 0,
                        is_training = True,
                        causality = False,
                        scope = "mulithead_attention",
                        reuse = None):
    with tf.variable_scope(scope,reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projection
        Q = tf.layers.dense(queries,num_units,activation=tf.nn.relu) # [batch_size,T,num_units]
        K = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #
        V = tf.layers.dense(keys,num_units,activation=tf.nn.relu) #

        # Split and Concat
        Q_ = tf.concat(tf.split(Q,num_heads,axis=2),axis=0) # [batch_size*num_heads, T, num_units/num_heads]
        K_ = tf.concat(tf.split(K,num_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,num_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0,2,1]))     # [batch_size*num_heads, T, T]
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)   # 相当于注意力矩阵，每一行都是一个query和所有的key的相似性

        # 这里是对填充的部分进行一个mask，这些位置的attention score变为极小，我们的embedding操作中是有一个padding操作的，
        # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys,axis=-1)))     
        key_masks = tf.tile(key_masks,[num_heads,1])      # [batch_size*num_heads, T]
        key_masks = tf.tile(tf.expand_dims(key_masks,1),[1,tf.shape(queries)[1],1])     # [batch_size*num_heads, T, T]

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks,0),paddings,outputs)     #  [batch_size*num_heads, T, T],填充部分的列为0

        # 这里其实就是进行一个mask操作，不给模型看到未来的信息。(针对decoder)
        if causality:
            diag_vals = tf.ones_like(outputs[0,:,:])    # [T, T]
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(outputs)[0],1,1])    # [batch_size*num_heads, T, T]  下三角矩阵

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks,0),paddings,outputs)     # [batch_size*num_heads, T, T]  decoder的self-attention的注意力矩阵上三角部分全为0

        outputs = tf.nn.softmax(outputs)

        # Query Mask
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries,axis=-1)))
        query_masks = tf.tile(query_masks,[num_heads,1])
        query_masks = tf.tile(tf.expand_dims(query_masks,-1),[1,1,tf.shape(keys)[1]])
        outputs *= query_masks

        # Dropout
        outputs = tf.layers.dropout(outputs,rate = dropout_rate,training = tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs,V_)   # [batch_size*num_heads, T, num_units/num_units]
        # restore shape
        outputs = tf.concat(tf.split(outputs,num_heads,axis=0),axis=2)   # [batch_size, T, num_units]
        # Residual connection
        outputs += queries
        # Normalize
        outputs = normalize(outputs)
    return outputs
	
	
def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)    # [batch_size,T,2048]

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)    # # [batch_size,T,512]
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = normalize(outputs)
    return outputs


#######################################
# encoder结构


with tf.variable_scope("encoder"):
    # Embedding
    self.enc = embedding(self.x,
                         vocab_size=len(de2idx),
                         num_units = hp.hidden_units,
                         zero_pad=True, # 让padding一直是0
                         scale=True,
                         scope="enc_embed")

    ## Positional Encoding
    if hp.sinusoid:
        self.enc += positional_encoding(self.x,
                                        num_units = hp.hidden_units,
                                        zero_pad = False,
                                        scale = False,
                                        scope='enc_pe')

    else:
        self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]),0),[tf.shape(self.x)[0],1]),
                              vocab_size = hp.maxlen,
                              num_units = hp.hidden_units,
                              zero_pad = False,
                              scale = False,
                              scope = "enc_pe")
    # [batch_size, T, hidden_units]
	
    ##Drop out
    self.enc = tf.layers.dropout(self.enc,rate = hp.dropout_rate,
                                 training = tf.convert_to_tensor(is_training))

    ## Blocks
    for i in range(hp.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
            ### MultiHead Attention
            self.enc = multihead_attention(queries = self.enc,
                                           keys = self.enc,
                                           num_units = hp.hidden_units,
                                           num_heads = hp.num_heads,
                                           dropout_rate = hp.dropout_rate,
                                           is_training = is_training,
                                           causality = False
                                           )
            self.enc = feedforward(self.enc,num_units = [4 * hp.hidden_units,hp.hidden_units])


#############################################
# decoder结构

with tf.variable_scope("decoder"):
    # Embedding
    self.dec = embedding(self.decoder_inputs,
                         vocab_size=len(en2idx),
                         num_units = hp.hidden_units,
                         scale=True,
                         scope="dec_embed")

    ## Positional Encoding
    if hp.sinusoid:
        self.dec += positional_encoding(self.decoder_inputs,
                                        vocab_size = hp.maxlen,
                                        num_units = hp.hidden_units,
                                        zero_pad = False,
                                        scale = False,
                                        scope = "dec_pe")
    else:
        self.dec += embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
            vocab_size=hp.maxlen,
            num_units=hp.hidden_units,
            zero_pad=False,
            scale=False,
            scope="dec_pe")

    # Dropout
    self.dec = tf.layers.dropout(self.dec,
                                rate = hp.dropout_rate,
                                training = tf.convert_to_tensor(is_training))

    ## Blocks
    for i in range(hp.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
            ## Multihead Attention ( self-attention)
            self.dec = multihead_attention(queries=self.dec,
                                           keys=self.dec,
                                           num_units=hp.hidden_units,
                                           num_heads=hp.num_heads,
                                           dropout_rate=hp.dropout_rate,
                                           is_training=is_training,
                                           causality=True,
                                           scope="self_attention")

            ## Multihead Attention ( vanilla attention)
            self.dec = multihead_attention(queries=self.dec,
                                           keys=self.enc,
                                           num_units=hp.hidden_units,
                                           num_heads=hp.num_heads,
                                           dropout_rate=hp.dropout_rate,
                                           is_training=is_training,
                                           causality=False,
                                           scope="vanilla_attention")

            ## Feed Forward
            self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])
