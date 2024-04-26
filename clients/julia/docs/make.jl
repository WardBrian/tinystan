using Documenter, TinyStan
using DocumenterMarkdown

makedocs(
    format = Markdown(),
    repo = "https://github.com/WardBrian/TinyStan/blob/main{path}#{line}",
)

cp(
    joinpath(@__DIR__, "build/julia.md"),
    joinpath(@__DIR__, "../../../docs/languages/julia.md");
    force = true,
)
cp(
    joinpath(@__DIR__, "build/assets/Documenter.css"),
    joinpath(@__DIR__, "../../../docs/_static/css/Documenter.css");
    force = true,
)
