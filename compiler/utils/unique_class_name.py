count = {}
def unique_class_name(name):
    if not type(name)==str:
        name = type(name).__name__

    if name.startswith("Forward"):
        name = name.replace("Forward","F")
    elif name.startswith("Backward"):
        name = name.replace("Backward","B")

    if name in count:
        count[name] += 1 
    else:
        count[name] = 0

    return f"{name}_{count[name]}"

