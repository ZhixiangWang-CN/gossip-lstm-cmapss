# Experimental protocol

## Data and targets

The shipped triplet is the FD001 subset of the public NASA C-MAPSS release (see data/README.md); its SHA-256 hashes are checked at load time. Reference: [NASA C-MAPSS description](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

For training engines, RUL(t)=T_failure-t. For truncated test engines, RUL(t)=T_observed+terminal_RUL-t. The input window includes t; its target is 1[RUL(t)<=30]. Every valid end index is used, including the final window. No RUL or class label enters the feature vector. Twenty-five features are 3 settings, cycle_norm, and 21 sensors. The cycle ablation sets that input channel to zero, keeping the architecture and communication size constant.

Official train/test engines are distinct physical units even when their numeric IDs match. Within official training data, 80% of whole engines train and 20% validate. Seed-specific splits are shared across all paired methods. Normalization is fit only on training engines. No validation motor contributes to gradient updates or scaling. Test data are never used for threshold tuning. Min/max extrapolation can produce test values outside [0,1]; no clipping is applied.

The common normalizer is an offline preprocessing reference computed from pooled training-engine extrema. It can in principle be constructed with min/max statistic exchange, but that distributed preprocessing protocol and its traffic are NOT implemented or included in the communication totals. Likewise, a shared class-count statistic sets positive_weight for weighted BCE. Thus training data locality should not be described as an end-to-end deployed private pipeline.

IID means random whole-engine allocation, not exactly identical sample distributions. Lifetime non-IID means sorted whole-engine lifetime shards using training histories only. Validation/test engines are independently randomly allocated to nodes. This tests training heterogeneity with a common evaluation population; it is not a simulation of known site-specific test domains.

## Architecture and learning budget

Stacked LSTM 100/50; dropout .2 after each LSTM; sigmoid output via logits and BCEWithLogitsLoss. Adam lr=.001; batch=200; clip norm=5; 50 epochs; one local epoch per round. Optimizer moments reset each round for EVERY method, including centralized. PyTorch CPU execution is deterministic on the recorded environment; bitwise agreement across operating systems/library versions is not promised. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

PyTorch has two recurrent biases per gate. Here bias_hh is fixed to zero and excluded from learning/communication; only bias_ih is trained, with forget-gate bias initialized to one. There are 80,651 trainable/transmitted float32 parameters and 600 immutable zero bias entries. Common initialization is copied to every node. Input weights use Xavier; recurrent weights orthogonal initialization; dense weights Xavier.

All methods see the same training windows and one pass per epoch, except deliberately failed nodes. Centralized minibatch count differs from the sum of locally rounded minibatch counts; equal processed-example budget does not imply exactly equal optimizer steps. Local-only is essential: it measures whether communication adds anything. FedAvg uses sample-count weighting. Standard gossip uses uniform-node stationary weighting; mass_gossip explicitly tests matching the sample-mass objective. Positive-weight BCE uses clipped global training negative/positive ratio (maximum 20) and is compared in both FedAvg and gossip. No early stopping or best-epoch selection is used; final epoch is evaluated. Validation thresholds are secondary post-training policies.

## Synchronous execution and gossip

Nodes train sequentially in one CPU process. After all active nodes finish, parameter snapshots are taken, mixed once, and retained. No old local checkpoint is reloaded. The next round begins with the mixed parameters. This is synchronous simulated decentralized learning, not asynchronous hardware networking.

Ordinary Metropolis mixing on the currently successful undirected graph uses W_ij=1/(1+max(d_i,d_j)) and W_ii=1-sum_j W_ij. For a healthy ring this is self/left/right averaging by 1/3. Dropped exchanges preserve row/column sums through self-weight. Connectivity is measured indirectly via active-edge count and SLEM; disconnected graphs may have SLEM=1. Local SGD perturbs consensus each epoch, so convergence of pure averaging is not a proof of global neural-model consensus.

Mass-aware mixing uses W_ij=min(1/(1+d_i), m_j/(m_i(1+d_j))) with training-window mass m_i. It is row-stochastic and reversible with stationary mass proportional to m_i; it is generally not doubly stochastic. This is an established Metropolis-Hastings idea used as an objective-alignment hypothesis, not a new averaging theorem.

Event variants send a trigger control scalar (modeled 16 bytes per directed neighbor) each scheduled round. Node i triggers when ||theta_i-last_sent_i||/(||last_sent_i||+epsilon)>=.02 or silence reaches 5 rounds. An edge exchanges if either endpoint triggers. The node reference is its pre-mix parameter vector at the most recent successful incident exchange, not a separate per-neighbor cache. A maximum silence trigger still cannot guarantee delivery under faults. Sensitivity thresholds .01/.05 and periodic every-2/every-5 comparisons are preconfigured. Periodic communication should be compared on the communication–performance frontier; equal epochs do not imply equal byte budgets.

## Network scenarios

| Scenario | Defined intervention |
|---|---|
| none | All participants and links available |
| packet_loss | Independently simulated 20% directed message loss |
| delay | Each directed latency uniform 20–300 ms; messages beyond 100 ms deadline discarded |
| intermittent | Each node communication availability sampled with probability .7 per round; local training continues |
| node_failure | Node index min(7,N-1) stops computation and communication from half the epochs onward |
| server_outage | Known-unavailable FedAvg coordinator between 40% and 70% of training; local updates continue; P2P has no coordinator |

Gossip only applies edges when both directions arrive on time, preserving symmetric availability for standard mixing. This atomic-pair rule presumes reliable exchange coordination; ACK/barrier/retry bytes and transport headers are excluded. It is not a deployed network protocol. FedAvg aggregates successful uploads and updates nodes receiving successful downlinks. Failed/reconnected clients retain their last local model until a successful update. No repair, re-routing or asynchronous stale-update queue is implemented.

The all-node primary evaluation includes retained models for diagnosis, even if a node is unavailable for real service. A separate served_window_fraction and served_metrics report availability-filtered inference at the final round. A score from an offline retained model is not proof that the failed service remains available. A known server outage is intentionally asymmetric and only answers the coordinator-dependence question, not general fault superiority.

Attempted bytes, delivered bytes and applied-message flags are separate. Payload is 322,604 bytes. Ring has 20 directed model messages per healthy round at N=10; FedAvg has 10 up + 10 down. Fifty rounds yield 322,604,000 bytes = 307.659 MiB, not MB. Event trigger bytes are included. Initial model provisioning, scaler/class statistics, headers, serialization and ACKs are excluded and must be stated.

training_wall_s includes sequential local training and mixing, excluding periodic evaluation, plotting and checkpoint I/O. It is NOT end-to-end runtime. Modeled parallel round time is max(local CPU times)+modeled network time, assuming homogeneous independent CPUs and configured bandwidth/latency. It excludes contention and central aggregation CPU overhead; do not call it observed acceleration. Measuring algorithm wall time on a single laptop cannot substantiate asynchronous deployment speedup.

## Metrics and inference

Primary endpoint: positive-class F1 on one terminal window per eligible test engine at threshold .5 (93 engines for this dataset). Secondary: all 8,255 windows. Always report the observation population and positive prevalence. Three policies: fixed .5; pooled-validation F1-maximizing threshold; pooled-validation threshold maximizing recall subject to FPR<=.02 on a fixed .01–.99 grid. The latter is a validation constraint, not a test guarantee. No feasible grid threshold -> 1.000001, meaning no alarms.

Pooled positive-class F1 is 2TP/(2TP+FP+FN). This is not sklearn's micro-average over both binary classes (which equals accuracy). Node-macro and class-macro are distinct and must be named. Node tables support separately computing node means; main results are pooled. F1 is zero when its denominator is zero; ROC-AUC is undefined for a single-class cohort. Average precision is reported as AP, not silently labeled trapezoidal PR-AUC. Confidence intervals use engine-cluster resampling, not independent window resampling. Replicates with no positive labels are excluded from the common CI cohort and their counts recorded; this conditional limitation is material for very rare events.

Seed-level means, sample SDs and t-intervals are separate from engine-bootstrap intervals. Scalar t-intervals may extend outside [0,1] and are intentionally not clipped. The predeclared paired terminal-F1 comparisons use exact two-sided sign-flip tests (up to 16 pairs), Holm-adjusted across comparisons in the generated report. Five pairs have minimum possible p=.0625, so ten seeds may be needed for inferential claims. Nonsignificance is not noninferiority; no noninferiority margin was specified here.

Common validation-probe predictions and parameter distances diagnose consensus. No raw common test sample is broadcast during training. This evaluation probe is an offline analysis resource; it is not a necessary training dependency. Loss-threshold membership diagnostic aggregates mean negative loss over each engine's last up-to-50 windows for training members and validation nonmembers. Class/trajectory differences can confound it; no shadow-model tuning, DP budget, gradient-inversion attack or comprehensive privacy proof is provided.

## Limits on generalization

Only FD001 ships. External triplets can be evaluated using the CLI. N=5/10/20/40 changes the same total training population, so it is fixed-data strong-scaling/fragmentation, not weak scaling or a large industrial deployment. At N=40 there are roughly two training engines per node. This can expose degradation, but cannot justify arbitrary scalability. Real industrial validation remains outstanding.
