def early_stopping(loss_pred, patience):
	if not hasattr(early_stopping, "best_loss"):
		early_stopping.best_loss = 1
	if not hasattr(early_stopping, "patience_count"):
		early_stopping.patience_count = 0

	if loss_pred <= early_stopping.best_loss:
		early_stopping.best_loss = loss_pred
		early_stopping.patience_count = 0
	else:
		early_stopping.patience_count += 1

	if early_stopping.patience_count > patience:
		return 1
	else:
		return 0

