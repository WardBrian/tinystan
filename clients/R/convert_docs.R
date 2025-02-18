# Converts R documentation (.Rd) files to markdown (.md) files for use in
# Sphinx.

library(rd2markdown)
library(roxygen2)

roxygen2::roxygenize()

files <- list.files("man", pattern = "*.Rd")

for (f in files) {
    name <- substr(f, 1, nchar(f) - 3)

    # read .Rd file and convert to markdown
    rd <- rd2markdown::get_rd(file = file.path(".", "man", f))
    md <- rd2markdown::rd2markdown(rd, fragments = c())
    # replaces the headers with more appropriate levels for embedding
    md_indented <- gsub("(#+)", "\\1##", md)

    # write it to the docs folder
    writeLines(md_indented, file.path("..", "..", "docs", "languages", "_r",
        paste0(name, ".md")))

}
