# feat/ssm-knowledge-distillation

### Short description

Perform classic knowledge distillation to train a smaller SSM, do this for different distillation hyperparamters to layout the relation between those parameters and the robustness of the model.

### Long description

Objective: Systematically map how Knowledge Distillation (KD) hyperparameters (temperature, loss weighting) affect not just the accuracy, but the robustness of a miniaturized student SSM.Methodology: Set up a teacher-student KD pipeline. Run a grid search over key KD hyperparameters: Temperature (T \in [1, 2, 5, 10]) and the balance factor (\alpha) between hard labels and soft teacher logits. Train multiple student SSMs under these different configurations.Evaluation & Metrics: For every student model trained, evaluate standard validation metrics. Crucially, subject all students to a robustness suite (adversarial attacks, noise injection, sequence truncation). Plot the correlation between KD hyperparameters (e.g., high temperature) and robustness scores.Expected Challenges: Finding the optimal temperature that softens the teacher's distribution enough to transfer hidden structural knowledge without flattening the signal so much that the student becomes overly sensitive to noise.