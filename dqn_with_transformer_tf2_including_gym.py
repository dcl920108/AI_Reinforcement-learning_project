import gym  # 引入OpenAI Gym库，用于创建和交互强化学习环境。Import OpenAI Gym for RL environment management.

import tensorflow as tf  # 引入TensorFlow深度学习框架。Import TensorFlow for deep learning.
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Input  # 导入Transformer相关层。Import Transformer-related layers.
from tensorflow.keras.models import Model  # 导入Keras模型基类。Import Keras Model base class.


class TransformerBlock(tf.keras.layers.Layer):
    # Transformer 块：多头注意力 + 前馈网络。Transformer block: multi-head attention + feed-forward network.
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)  # 多头注意力层。Multi-head attention layer.
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),  # 前馈隐藏层。Feed-forward hidden layer.
            Dense(embed_dim),  # 投影回嵌入维度。Project back to embedding dimension.
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # 层归一化1。Layer normalization 1.
        self.layernorm2 = LayerNormalization(epsilon=1e-6)  # 层归一化2。Layer normalization 2.
        self.dropout1 = tf.keras.layers.Dropout(rate)  # Dropout1。Dropout 1.
        self.dropout2 = tf.keras.layers.Dropout(rate)  # Dropout2。Dropout 2.

    def call(self, inputs, training):
        # 前向计算：注意力 + 残差 + 归一化 + 前馈 + 残差 + 归一化。
        # Forward pass: attention + residual + norm + feed-forward + residual + norm.
        attn_output = self.att(inputs, inputs)  # 自注意力。Self-attention.
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # 第一个残差连接与归一化。First residual connection + normalization.
        ffn_output = self.ffn(out1)  # 前馈网络输出。Feed-forward output.
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # 第二个残差连接与归一化。Second residual connection + normalization.


class DQNAgent:
    # DQNAgent：结合Transformer用于Q学习的智能体。DQN agent with Transformer for Q-learning.
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 状态维度。State dimension.
        self.action_size = action_size  # 动作数目。Number of actions.
        self.memory = []  # 经验回放。Experience replay memory.
        self.gamma = 0.99  # 折扣因子。Discount factor.
        self.epsilon = 1.0  # 探索率初始值。Initial exploration rate.
        self.epsilon_min = 0.1  # 最小探索率。Minimum exploration rate.
        self.epsilon_decay = 0.995  # 探索率衰减。Exploration rate decay.
        self.batch_size = 64  # 批量大小。Batch size.
        self.learning_rate = 1e-3  # 学习率。Learning rate.
        self.model = self.build_model()  # 构建主网络。Build the main network.
        self.target_model = self.build_model()  # 构建目标网络。Build the target network.
        self.update_target_model()  # 初始化同步。Initialize target network.

    def build_model(self):
        # 构建含TransformerBlock的DQN模型。Build DQN model with TransformerBlock.
        inputs = Input(shape=(self.state_size,))  # 输入层。Input layer.
        x = tf.expand_dims(inputs, axis=1)  # 增加序列维度，便于Transformer处理。Add sequence dim for Transformer.
        x = TransformerBlock(embed_dim=self.state_size, num_heads=2, ff_dim=128)(x)  # Transformer块。Transformer block.
        x = tf.keras.layers.Flatten()(x)  # 展平输出。Flatten output.
        x = Dense(128, activation='relu')(x)  # 全连接层。Dense layer.
        outputs = Dense(self.action_size, activation='linear')(x)  # 输出Q值。Output Q-values.
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')  # 编译模型。Compile model.
        return model

    def update_target_model(self):
        # 同步主网络权重到目标网络。Sync main model weights to target model.
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # 存储经验到记忆。Store experience in memory.
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 根据epsilon-greedy策略选择动作。Choose action with epsilon-greedy.
        if tf.random.uniform([]) < self.epsilon:
            return gym.spaces.Discrete(self.action_size).sample()  # 随机动作。Random action.
        q_values = self.model.predict(state[None, :])
        return tf.argmax(q_values[0]).numpy()

    def replay(self):
        # 从记忆中抽样并训练网络。Sample from memory and train.
        if len(self.memory) < self.batch_size:
            return
        batch = tf.random.shuffle(self.memory)[:self.batch_size]
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # 计算目标Q值。Compute target Q-value.
                t = self.target_model.predict(next_state[None, :])[0]
                target += self.gamma * tf.reduce_max(t)
            target_f = self.model.predict(state[None, :])[0]
            target_f[action] = target
            states.append(state)
            targets.append(target_f)
        self.model.fit(tf.stack(states), tf.stack(targets), epochs=1, verbose=0)
        # 衰减epsilon。Decay epsilon.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('CartPole-v1')  # 创建环境。Create environment.
    state_size = env.observation_space.shape[0]  # 状态维度。State dimension.
    action_size = env.action_space.n  # 动作数目。Number of actions.
    agent = DQNAgent(state_size, action_size)  # 初始化智能体。Initialize agent.
    episodes = 500  # 训练轮数。Number of episodes.

    for e in range(episodes):
        state = env.reset()  # 重置环境。Reset environment.
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)  # 选择动作。Select action.
            next_state, reward, done, _ = env.step(action)  # 执行动作。Take action.
            agent.remember(state, action, reward, next_state, done)  # 存储记忆。Store memory.
            state = next_state
            total_reward += reward
            agent.replay()  # 训练网络。Train network.
        agent.update_target_model()  # 更新目标网络。Update target network.
        print(f"Episode: {e}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")  # 日志输出。Logging.
