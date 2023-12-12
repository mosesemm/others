package za.co.mashandiro.strutsfirst.service;

import java.util.ArrayList;
import java.util.List;

public class RegisterService {

    public List<String> retrieveSports() {
        List<String> sports = new ArrayList<String>();
        sports.add("Football");
        sports.add("Cricket");
        sports.add("Tennis");
        sports.add("Basketball");
        sports.add("Volleyball");
        return sports;
    }
}
