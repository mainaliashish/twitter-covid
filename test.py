# save numpy array as csv file
from numpy import asarray
from numpy import savetxt

# define data
data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to csv file
savetxt('data.csv', data, delimiter='', header='index')
