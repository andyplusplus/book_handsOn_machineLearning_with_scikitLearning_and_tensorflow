from datetime import datetime
log_dir_index = 0


def show_plt(plt, is_plt_show=False):
    if is_plt_show:
        show_plt(plt, is_plt_show=False)
    else:
        pass

def get_log_dir(root="tf_logs"):
    global log_dir_index
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/tf_logs/run-{}-{}/".format(root, now, log_dir_index)
    log_dir_index += 1
    return log_dir