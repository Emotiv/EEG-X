# Automatic final-test charts

History stores metric events. Summary stores the final value for a run. The earlier logger wrote scalar test scores only to Summary, so automatic workspaces discovered the confusion image but omitted the score charts.

The training script now publishes a final test event after checkpoint selection and retains the same scores in Summary. Fine-tuning and standalone linear probing share this reporting method. Test evaluation remains separate from the training and validation loop.

| Verification | Result |
|---|---:|
| Final score history entries in reference Neuralbench run | 1 |
| Final score history entries in old EEG-X run | 0 |
| Final score history entries with the new logger | 1 |
| Automatically generated test charts in a fresh project | 12 |
| Scalar charts / confusion matrix | 11 / 1 |
| Manual panel setup operations | 0 |
| Focused training, selection and workspace checks | 23 passed |
| Existing review runs preserved | 5 |

The online browser check used existing saved final-test predictions in a temporary private project. That project was removed after verification. New runs using the updated script can use the default Automated workspace without adding panels manually.
