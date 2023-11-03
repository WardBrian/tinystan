function get_ffistan_path()
    path = get(ENV, "FFISTAN", "")
    if path == ""
        # TODO artifact
        error("FFISTAN environment variable not set")
    end
    return path
end
