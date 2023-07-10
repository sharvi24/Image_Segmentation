class Util():
    """ Util class with util functions """
    def __init__(self) -> None:
        pass

    def compute_cumulative_sum(self, array: list) -> list:
        """Compute the cumulative sum of the input list

        Args:
            array (list): input array of numbers

        Returns:
            list: cummulative sum
        """
        N = len(array)
        cumsum = [0] * (N + 1)
        for i in range(N):
            cumsum[i+1] = cumsum[i] + array[i]
        return cumsum

    def compute_smoothed_output(self, array: list, cum_array: list, window_size: int) -> list:
        """Compute the smoothed output list using the cumulative sum

        Args:
            array (list): input array
            cum_array (list): cummulative sum array
            window_size (int): smoothing window size

        Returns:
            list: smoothed output list
        """
        N = len(array)
        outlist = [0] * N
        for i in range(N):
            kmin = max(0, i - window_size)
            kmax = min(N - 1, i + window_size)
            count = kmax - kmin + 1
            outlist[i] = (cum_array[kmax+1] - cum_array[kmin]) / count
        return outlist


# smooth function
def smooth(inlist, h):
    """
    Performs a basic smoothing of an input list and returns the result.

    Parameters:
        inlist (list): The input list to be smoothed with each item of 'float' type.
        h (int): The smoothing window size.

    Returns:
        list: The smoothed output list.
    """
    util = Util()
    print(inlist)
    cum_array = util.compute_cumulative_sum(array=inlist)
    outlist = util.compute_smoothed_output(
        array=inlist,
        cum_array=cum_array,
        window_size=h
    )
    return outlist
