using Downloads, Tar, TOML, Inflate

"""
Windows-friendly way to get the user's home directory.
"""
function get_home()
    if Sys.iswindows()
        if haskey(ENV, "USERPROFILE")
            userhome = ENV["USERPROFILE"]
        elseif !haskey(ENV, "HOMEPATH")
            userhome = path
        else
            drive = get(ENV, "HOMEDRIVE", "")
            userhome = joinpath(drive, ENV["HOMEPATH"])
        end
        return userhome
    else
        return expanduser("~/")
    end
end

function get_version()
    return VersionNumber(
        TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))["version"],
    )
end
const pkg_version = get_version()

const HOME_TINYSTAN = joinpath(get_home(), ".tinystan")
const CURRENT_TINYSTAN = joinpath(HOME_TINYSTAN, "tinystan-$pkg_version")

const RETRIES = 5


function get_tinystan_src()

    url =
        "https://github.com/WardBrian/tinystan/releases/download/" *
        "v$pkg_version/tinystan-$pkg_version.tar.gz"
    mkpath(HOME_TINYSTAN)
    tmp = nothing
    err_text = "Failed to download TinyStan $pkg_version from github.com."
    for i = 1:RETRIES
        try
            tmp = Downloads.download(url)
            break
        catch
            if i == RETRIES
                error(err_text)
            end
            println(err_text)
            println("Retrying ($(i+1)/$RETRIES)...")
            sleep(1)
        end
    end

    try
        tmp_extr = Tar.extract(IOBuffer(Inflate.inflate_gzip(tmp)), copy_symlinks = true)
        mv(joinpath(tmp_extr, "tinystan-$pkg_version"), CURRENT_TINYSTAN)
    catch
        error("Failed to unpack $tmp during installation")
    end
end
