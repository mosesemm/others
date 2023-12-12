package za.co.mashandiro.strutsfirst.model;

public class Person {
    private String firstName;
    private String lastName;
    private String email;
    private String favoriteSport;
    private String gender;
    private int age;
    private boolean over21;

    public String getFirstName() {
        return firstName;
    }
    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }
    public String getLastName() {
        return lastName;
    }
    public void setLastName(String lastName) {
        this.lastName = lastName;
    }
    public String getEmail() {
        return email;
    }
    public void setEmail(String email) {
        this.email = email;
    }
    public int getAge() {
        return age;
    }
    public void setAge(int age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "First Name: " + getFirstName() + " Last Name:  " + getLastName() + 
        " Email:      " + getEmail() + " Age:      " + getAge() + " Sport:  "+ getFavoriteSport();
    }
    public String getFavoriteSport() {
        return favoriteSport;
    }
    public void setFavoriteSport(String favoriteSport) {
        this.favoriteSport = favoriteSport;
    }
    public String getGender() {
        return gender;
    }
    public void setGender(String gender) {
        this.gender = gender;
    }
    public boolean isOver21() {
        return over21;
    }
    public void setOver21(boolean over21) {
        this.over21 = over21;
    }
}
