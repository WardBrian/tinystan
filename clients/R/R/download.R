current_version <- packageVersion("tinystan")
current_version_list <- list(
  major = current_version$major,
  minor = current_version$minor,
  patch = current_version$patch
)
HOME_TINYSTAN <- path.expand(file.path("~", ".tinystan"))
CURRENT_TINYSTAN <- file.path(
  HOME_TINYSTAN,
  paste0("tinystan-", current_version)
)

RETRIES <- 5

get_tinystan_src <- function() {
  url <- paste0(
    "https://github.com/WardBrian/tinystan/releases/download/",
    "v",
    current_version,
    "/tinystan-",
    current_version,
    ".tar.gz"
  )

  dir.create(HOME_TINYSTAN, showWarnings = FALSE, recursive = TRUE)
  temp <- tempfile()
  err_text <- paste(
    "Failed to download TinyStan",
    current_version,
    "from github.com."
  )
  for (i in 1:RETRIES) {
    tryCatch(
      {
        download.file(
          url,
          destfile = temp,
          mode = "wb",
          quiet = TRUE,
          method = "auto"
        )
      },
      error = function(e) {
        cat(err_text, "\n")
        if (i == RETRIES) {
          stop(err_text, call. = FALSE)
        } else {
          cat("Retrying (", i + 1, "/", RETRIES, ")...\n", sep = "")
          Sys.sleep(1)
        }
      }
    )
  }

  tryCatch(
    {
      untar(temp, exdir = HOME_TINYSTAN)
    },
    error = function(e) {
      stop(paste("Failed to unpack", url, "during installation"), call. = FALSE)
    }
  )

  unlink(temp)
}
