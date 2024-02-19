package za.co.mashandiro.springplay.controllers;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RestController;

import za.co.mashandiro.springplay.model.Customer;
import za.co.mashandiro.springplay.services.CustomerService;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.extern.log4j.Log4j2;



@Log4j2
@RestController
public class CustomerResource {

    @Autowired
    private CustomerService customerService;

    @RequestMapping(path = "/customers", method=RequestMethod.GET)
    public List<Customer> getMethodName() {
        return customerService.getCustomers();
    }

    @RequestMapping(path = "/customers1", method=RequestMethod.GET)
    public String getMethodName1() throws JsonProcessingException {
        log.info("and also this is called...");
        return new ObjectMapper().writeValueAsString(customerService.getCustomers());
    }

    @RequestMapping(path = "/customers2", method=RequestMethod.GET)
    public String getMethodName2() {
        log.info("and also this is called...getMethodName2");
        String results = null;
        try{
            results = new ObjectMapper().writeValueAsString(customerService.getCustomersSp());
        }
        catch(JsonProcessingException e){
            log.error("Error occurred while processing the request", e);
        }

        return results;
    }


    

}
