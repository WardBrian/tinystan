
function get_make()
    get(ENV, "MAKE", "make")
end

function validate_stan_dir(path::AbstractString)
    if !isdir(path)
        error("Path does not exist!\n$path")
    end
    if !isfile(joinpath(path, "Makefile"))
        error(
            "Makefile does not exist at path! Make sure it was installed correctly.\n$path",
        )
    end
end

"""
    set_tinystan_path!(path)

Set the path TinyStan.
"""
function set_tinystan_path!(path::AbstractString)
    validate_stan_dir(path)
    ENV["TINYSTAN"] = path
end

"""
    get_tinystan_path() -> String

Return the path the the TinyStan directory.

If the environment variable `TINYSTAN` is set, this will be returned.
Otherwise, this function downloads a matching version of TinyStan under
a folder called `.tinystan` in the user's home directory.

See [`set_tinystan_path!()`](@ref) to set the path from within Julia.
"""
function get_tinystan_path()
    path = get(ENV, "TINYSTAN", "")
    if path == ""
        path = CURRENT_TINYSTAN
        try
            validate_stan_dir(path)
        catch
            println(
                "TinyStan not found at location specified by \$TINYSTAN " *
                "environment variable, downloading version $pkg_version to $path",
            )
            get_tinystan_src()
            num_files = length(readdir(HOME_TINYSTAN))
            if num_files >= 5
                @warn "Found $num_files different versions of TinyStan in $HOME_TINYSTAN. " *
                      "Consider deleting old versions to save space."
            end
            println("Done!")
        end
    end
    return path
end

"""
    compile_model(stan_file; stanc_args=[], make_args=[])

Run TinyStan’s Makefile on a `.stan` file, creating the `.so` used by StanModel and
return a path to the compiled library.
Arguments to `stanc3` can be passed as a vector, for example `["--O1"]` enables level 1 compiler
optimizations.
Additional arguments to `make` can be passed as a vector, for example `["STAN_NO_RANGE_CHECKS=true"]`
will disable bounds checking in the Stan Math library. If the same flags are defined in `make/local`,
the versions passed here will take precedent.

This function checks that the path to TinyStan is valid and will error if it is not.
This can be set with [`set_tinystan_path!()`](@ref).
"""
function compile_model(
    stan_file::AbstractString;
    stanc_args::AbstractVector{String} = String[],
    make_args::AbstractVector{String} = String[],
)
    tinystan = get_tinystan_path()
    validate_stan_dir(tinystan)

    if !isfile(stan_file)
        throw(SystemError("Stan file not found: $stan_file"))
    end
    if splitext(stan_file)[2] != ".stan"
        error("File '$stan_file' does not end in .stan")
    end

    absolute_path = abspath(stan_file)
    output_file = splitext(absolute_path)[1] * "_model.so"

    cmd = Cmd(
        `$(get_make()) $make_args "STANCFLAGS=--include-paths=. $stanc_args" $output_file`,
        dir = abspath(tinystan),
    )
    out = IOBuffer()
    err = IOBuffer()
    is_ok = success(pipeline(cmd; stdout = out, stderr = err))
    if !is_ok
        error(
            "Compilation failed!\nCommand: $cmd\nstdout: $(String(take!(out)))\nstderr: $(String(take!(err)))",
        )
    end
    return output_file
end

WINDOWS_PATH_SET = Ref{Bool}(false)

function tbb_found()
    try
        run(pipeline(`where.exe tbb.dll`, stdout = devnull, stderr = devnull))
    catch
        return false
    end
    return true
end

function windows_dll_path_setup()
    if Sys.iswindows() && !(WINDOWS_PATH_SET[])
        if tbb_found()
            WINDOWS_PATH_SET[] = true
        else
            # add TBB to %PATH%
            ENV["PATH"] =
                joinpath(get_tinystan_path(), "stan", "lib", "stan_math", "lib", "tbb") *
                ";" *
                ENV["PATH"]
            WINDOWS_PATH_SET[] = tbb_found()
        end
    end
end
