current_version <- packageVersion("ffistan")
HOME_FFISTAN <- path.expand(file.path("~", ".ffistan"))
CURRENT_FFISTAN <- file.path(HOME_FFISTAN, paste0("ffistan-", current_version))

RETRIES <- 5

get_ffistan_src <- function() {
    url <- paste0("https://github.com/WardBrian/ffistan/releases/download/", "v",
        current_version, "/ffistan-", current_version, ".tar.gz")

    dir.create(HOME_FFISTAN, showWarnings = FALSE, recursive = TRUE)
    temp <- tempfile()
    err_text <- paste("Failed to download FFIStan", current_version, "from github.com.")
    for (i in 1:RETRIES) {
        tryCatch({
            download.file(url, destfile = temp, mode = "wb", quiet = TRUE, method = "auto")
        }, error = function(e) {
            cat(err_text, "\n")
            if (i == RETRIES) {
                stop(err_text, call. = FALSE)
            } else {
                cat("Retrying (", i + 1, "/", RETRIES, ")...\n", sep = "")
                Sys.sleep(1)
            }
        })
    }

    tryCatch({
        untar(temp, exdir = HOME_FFISTAN)
    }, error = function(e) {
        stop(paste("Failed to unpack", url, "during installation"), call. = FALSE)
    })

    unlink(temp)
}
