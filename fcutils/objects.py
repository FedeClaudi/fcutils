# ---------------------------------------------------------------------------- #
#                              LISTS MANIPULATION                              #
# ---------------------------------------------------------------------------- #


def flatten_list(lst):
    """
    Flattens a list of lists
    
    :param lst: list

    """
    flatten = []
    for item in lst:
        if isinstance(item, list):
            flatten.extend(item)
        else:
            flatten.append(item)
    return flatten


def is_any_item_in_list(L1, L2):
    """
    Checks if an item in a list is in another  list

    :param L1: 
    :param L2: 

    """
    # checks if any item of L1 is also in L2 and returns false otherwise
    inboth = [i for i in L1 if i in L2]
    if inboth:
        return True
    else:
        return False


def calc_prob_item_in_list(ls, it):
    """[Calculates the frequency of occurences of item in list]
	
	Arguments:
		ls {[list]} -- [list of items]
		it {[int, array, str]} -- [items]
	"""

    n_items = len(ls)
    n_occurrences = len([x for x in ls if x == it])
    return n_occurrences / n_items
