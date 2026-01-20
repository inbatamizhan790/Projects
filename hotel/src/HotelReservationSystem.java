import java.util.ArrayList;
import java.util.Scanner;

public class HotelReservationSystem {

    static ArrayList<Room> rooms = new ArrayList<>();
    static ArrayList<Reservation> reservations = new ArrayList<>();

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Predefined rooms
        rooms.add(new Room(101, "Single", 1500));
        rooms.add(new Room(102, "Double", 2500));
        rooms.add(new Room(103, "Deluxe", 4000));

        int choice;

        do {
            System.out.println("\n--- HOTEL RESERVATION SYSTEM ---");
            System.out.println("1. View Available Rooms");
            System.out.println("2. Book Room");
            System.out.println("3. Cancel Booking");
            System.out.println("4. Exit");
            System.out.print("Enter choice: ");
            choice = sc.nextInt();

            switch (choice) {
                case 1:
                    viewAvailableRooms();
                    break;

                case 2:
                    bookRoom(sc);
                    break;

                case 3:
                    cancelBooking(sc);
                    break;

                case 4:
                    System.out.println("Thank you for using the system!");
                    break;

                default:
                    System.out.println("Invalid choice!");
            }
        } while (choice != 4);

        sc.close();
    }

    // View Available Rooms
    public static void viewAvailableRooms() {
        System.out.println("\nAvailable Rooms:");
        for (Room room : rooms) {
            if (!room.isBooked()) {
                System.out.println(room);
            }
        }
    }

    // Book Room
    public static void bookRoom(Scanner sc) {
        System.out.print("Enter room number to book: ");
        int roomNo = sc.nextInt();
        sc.nextLine(); // consume newline
        System.out.print("Enter customer name: ");
        String name = sc.nextLine();

        for (Room room : rooms) {
            if (room.getRoomNumber() == roomNo && !room.isBooked()) {
                room.bookRoom();
                reservations.add(new Reservation(name, room));
                System.out.println("Room booked successfully!");
                return;
            }
        }
        System.out.println("Room not available!");
    }

    // Cancel Booking
    public static void cancelBooking(Scanner sc) {
        System.out.print("Enter room number to cancel booking: ");
        int roomNo = sc.nextInt();

        for (Reservation res : reservations) {
            if (res.getRoom().getRoomNumber() == roomNo) {
                res.getRoom().cancelBooking();
                reservations.remove(res);
                System.out.println("Booking cancelled successfully!");
                return;
            }
        }
        System.out.println("No reservation found!");
    }
}
