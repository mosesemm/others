package za.co.mashandiro.springplay.services;

import java.util.List;

import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Service;

import lombok.extern.log4j.Log4j2;
import za.co.mashandiro.springplay.model.Customer;

@Log4j2
@Service
public class CustomerService {

    public List<Customer> getCustomers() {
        log.info("This is being called now....");
        return List.of(
            new Customer(1L, "John", "Doe"),
            new Customer(2L, "Jane", "Doe")
        );
    }

    @PreAuthorize("hasRole('ADMIN')")
    public List<Customer> getCustomersSp() {
        log.info("This is being called now....");
        return List.of(
            new Customer(1L, "Lebo", "Doe"),
            new Customer(2L, "Thabo", "Doe")
        );
    }
}
