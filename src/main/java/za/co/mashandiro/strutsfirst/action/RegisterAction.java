package za.co.mashandiro.strutsfirst.action;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import com.opensymphony.xwork2.ActionSupport;

import za.co.mashandiro.strutsfirst.model.Person;
import za.co.mashandiro.strutsfirst.service.RegisterService;

public class RegisterAction extends ActionSupport{

    private Person personBean;
    private String[] sports = {"Football", "Cricket", "Tennis", "Basketball"};
    private String[] genders = {"Male", "Female", "Other"};
    private RegisterService registerService;

    public void setRegisterService(RegisterService registerService) {
        this.registerService = registerService;
    }

    public List<String> getSports() {
        //return Arrays.asList(sports);
        return registerService.retrieveSports();
    }

    public List<String> getGenders() {
        return Arrays.asList(genders);
    }

    public String execute() throws Exception {

        return SUCCESS;
    }

    public Person getPersonBean() {
        return personBean;
    }

    public void setPersonBean(Person personBean) {
        this.personBean = personBean;
    }

    public void validate() {
        if(Optional.ofNullable(personBean.getFirstName()).orElse("").length() == 0) {
            addFieldError("personBean.firstName", "First name is required.");
        }
        if(personBean.getEmail().length() == 0) {
            addFieldError("personBean.email", "Email is required.");
        }
        if(personBean.getAge() < 18) {
            addFieldError("personBean.age", "Age is required and must be 18 or older plz.");
        }
    }
    
}
