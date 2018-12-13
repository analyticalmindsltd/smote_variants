library(reticulate)

python_path <- file.path(file.expand('~'), 'anaconda3', 'bin', 'python')
virtualenv_name <- 'base'

use_python(python_path)
use_virtualenv(virtualenv_name)

imbd <- import("imbalanced_databases")
sv <- import("smote_variants")

data <- imbalanced_databases$load_iris0()
sv$SMOTE()$sample(data$data, data$target)
