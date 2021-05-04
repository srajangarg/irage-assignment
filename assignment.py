import os
import pandas as pd
from sortedcontainers import SortedDict
from pandas.api.types import CategoricalDtype

class OrderStatus:
    def __init__(self, price=None, qty=None, side=None, security_id=None):
        self.price       = price
        self.qty         = qty
        self.side        = side
        self.security_id = security_id

    def active(self):
        # order has been sent to the exchange and has not been completely traded
        return (self.qty is not None and self.qty != 0)

    # update _this_ order status based on new event, return new updated status
    def update(self, update_dict):
        if   update_dict['Type'] == "N":
            return OrderStatus(update_dict['price'], update_dict['qty'],
                               update_dict['Side'], update_dict['SecurityId'])
        elif update_dict['Type'] == 'M':
            return OrderStatus(update_dict['price'], update_dict['qty'],
                               self.side, self.security_id)
        elif update_dict['Type'] == 'X':
            return OrderStatus(qty=0)
        elif update_dict['Type'] == 'T':
            return OrderStatus(update_dict['price'], self.qty - update_dict['qty'],
                               self.side, self.security_id)
        else:
            assert(False, "Invalid Type")

class OrderTimeline:
    # class to store the status of one order over it's life-cycle
    def __init__(self, order_tbt_df):
        # an order's lifecycle is stored as a sorteddict of timestamp -> OrderStatus
        self.time_status_map = SortedDict({0 : OrderStatus()})

        # iterating over the events involving this order
        for ix, row in order_tbt_df.iterrows():
            update_dict = row.to_dict()
            last_status = self.time_status_map.values()[-1]
            new_status  = last_status.update(update_dict)

            # add new order status to map
            self.time_status_map[update_dict['ExchTstamp']] = new_status

    # get the status of the order at given timestamp
    def status(self, timestamp):
        # return the largest value which is <= timestamp present in our map
        # http://www.grantjenks.com/docs/sortedcontainers/sortedlist.html#sortedcontainers.SortedList.bisect_right
        return self.time_status_map.values()[
                self.time_status_map.bisect_right(timestamp)-1]

class OrderManager:
    # class to manage one client's orders
    def __init__(self, tbt_df, exch_ids_df):
        self.tbt_df   = tbt_df
        self.exch_ids = list(exch_ids_df["0"])
        self.populate_order_timelines()

    # uses the tick-by-tick data to populate each order's lifecycle
    def populate_order_timelines(self):
        # all the orders are stored as a dict of exch_id -> OrderTimeline
        self.order_timeline_map = {}
        for exch_id in self.exch_ids:
            # print(exch_id)
            # filtered df containing information about exch_id only
            order_tbt_df = self.tbt_df[(self.tbt_df['ExchId1'] == exch_id) |
                                       (self.tbt_df['ExchId2'] == exch_id)]
            # adding the exch_id and corresponding timeline to our map
            self.order_timeline_map[exch_id] = OrderTimeline(order_tbt_df)

    # returns all the active orders we are managing at a given timestamp
    def get_active_orders(self, timestamp):
        # iterate over each exch_id and add it to the list of active orders
        # based on its `active` status
        active_orders = []
        for exch_id in self.exch_ids:
            order_status = self.order_timeline_map[exch_id].status(timestamp)
            if order_status.active():
                active_orders.append({
                    "timestamp":   timestamp,
                    "price":       order_status.price,
                    "qty":         order_status.qty,
                    "side":        order_status.side,
                    "security_id": order_status.security_id,
                })

        return active_orders

def get_all_orders(tbt_data_file_path, exch_ids_file_path, timestamp_file_path):
    tbt        = pd.read_csv(tbt_data_file_path)
    exch_ids   = pd.read_csv(exch_ids_file_path)
    timestamps = pd.read_csv(timestamp_file_path, header=None)[0]

    # defining an ordering on trade type and subsequently sorting
    cat_size_order = CategoricalDtype(['N', 'M', 'X', 'T'], ordered=True)
    tbt['Type']    = tbt['Type'].astype(cat_size_order)
    tbt            = tbt.sort_values(['ExchTstamp', 'Type'])

    om            = OrderManager(tbt, exch_ids)
    active_orders = []

    for timestamp in timestamps:
        active_orders += om.get_active_orders(timestamp)

    active_orders_df = pd.DataFrame(active_orders, columns=['timestamp', 'price', 'qty', 'side',
                                                            'security_id']).set_index('timestamp')
    active_orders_df.to_csv("output.csv")
    print(f"File saved at {os.path.join(os.getcwd(), 'output.csv')}")

if __name__ == "__main__":
    tbt_data_file_path  = "tbt_data_UCB/tbt_data_UCB.csv"
    exch_ids_file_path  = "tbt_data_UCB/orders_exch_ids.csv"
    timestamp_file_path = "tbt_data_UCB/timestamp_file.csv"
    get_all_orders(tbt_data_file_path, exch_ids_file_path, timestamp_file_path)
