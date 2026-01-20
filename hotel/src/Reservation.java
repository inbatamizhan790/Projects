public class Reservation {
    private String customerName;
    private Room room;

    public Reservation(String customerName, Room room) {
        this.customerName = customerName;
        this.room = room;
    }

    public String getCustomerName() {
        return customerName;
    }

    public Room getRoom() {
        return room;
    }

    @Override
    public String toString() {
        return "Customer: " + customerName +
                ", Room No: " + room.getRoomNumber();
    }
}
