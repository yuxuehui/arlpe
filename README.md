# Basal Glucose Control in Type 1 Diabetes Using An Off-policy Meta Reinforcement Learning Framework with Active Learning

\begin{algorithm}
	\caption{Meta-training}
	\label{meta_training}
	\begin{algorithmic}[1]  %1表示每隔一行编号	
		\Require Batch of training tasks $\mathcal{T}_{i=1...T} \sim p(\mathcal{T})$, learning rates $\tau_1$, $\tau_2$, $\tau_3$, $\tau_4$, $\tau_5$. 
        \State Initialize replay buffers $\mathcal{B}^i$ for each training task 
        \State Initialize parameter vectors $\psi$, $\bar{\psi}$, $\phi$, $\theta$, $\xi$
		\While {not done}
        \For {each $\mathcal{T}_i$}
		    \State Initialize context $\bm{c}^i= \emptyset$;
            \For{each environment step}
                \State Sample $\bm{z} \sim q_{\phi}(\bm{z}|\bm{c}^i)$
                \State $\bm{a}_t \sim \pi_{\theta}(\bm{a}_t|\bm{o}_t,\bm{z}_t)$
                \State $\bm{o}_{t+1} \sim p(\bm{o}_{t+1}|\bm{o}_t, \bm{a}_t)$
                \State Compute reward $r_t$ according to \eqref{eq22}
                \State Add $(\bm{o}_t,\bm{a}_t,\bm{o}_{t+1},r_t)$ to $\mathcal{B}^i$
                \State Update $\bm{c}^i=\left\{ (\bm{o}_j,\bm{a}_j,\bm{o}'_j,r_j)_{j:1...N} \right\} \sim \mathcal{B}^i$
            \EndFor
		\EndFor
        \For{step in training steps}
            \For{each $\mathcal{T}^i$}
                \State Sample context $\bm{c}^i \sim \mathcal{S}_c(\mathcal{B}^i)$ 
                \State Sample RL batch $b^i \sim \mathcal{B}^i$
                \State Sample $\bm{z} \sim q_{\phi}(\bm{z}|\bm{c}^i)$
                \State $\psi \leftarrow \psi - \tau_1 \nabla_{\psi}(\mathcal{L}_V)$
                \State $\xi_i \leftarrow \xi_i - \tau_2 \nabla_{\xi_i}(\mathcal{L}_Q)$ for $i \in \{1,2\}$
                \State $\theta \leftarrow \theta - \tau_3 \nabla_{\theta}(\mathcal{L}_{\pi})$
                \State $\phi \leftarrow \phi - \tau_4 \nabla_{\phi}(\mathcal{L}_{\phi})$
                \State $\bar{\psi} \leftarrow \tau_5\psi + (1-\tau_5)\bar{\psi}  $
            \EndFor
        \EndFor
        \EndWhile
	\end{algorithmic}
\end{algorithm}
