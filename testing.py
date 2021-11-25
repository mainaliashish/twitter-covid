# Python3 program to illustrate store
# efficiently using pickle module
# Module translates an in-memory Python object
# into a serialized byte stream—a string of
# bytes that can be written to any file-like object.

import pickle


def storeData():
    # initializing data to be stored in db
    Omkar = {'key': 'Omkar', 'name': 'Omkar Pathak',
             'age': 21, 'pay': 40000}
    Jagdish = {'key': 'Jagdish', 'name': 'Jagdish Pathak',
               'age': 50, 'pay': 50000}

    # database
    db = {}
    db['Omkar'] = Omkar
    db['Jagdish'] = Jagdish

    logger = {'key': ['Omkar', 'Jagdish']}

    # Its important to use binary mode
    # for item in db:
    #     print(item)
    #
    # exit(0)
    dbfile = open('examplePickle', 'wb')
    lfile = open('logger', 'wb')

    # source, destination
    pickle.dump(db, dbfile)
    pickle.dump(logger, lfile)
    dbfile.close()


def loadData():
    # for reading also binary mode is important
    lfile = open('logger', 'rb')
    dbfile = open('examplePickle', 'rb')
    lfile = pickle.load(lfile)
    dbfile = pickle.load(dbfile)
    print(lfile)
    print(dbfile)
    if lfile in dbfile:
        print(lfile)

    # print(db)
    # for keys in db:
    #     print(keys, '=>', db[keys])
    lfile.close()


if __name__ == '__main__':
    storeData()
    loadData()
