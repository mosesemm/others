package za.co.mashandiro.springplay.controllers;

import java.security.Principal;

import javax.servlet.http.HttpServletRequest;

import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.ModelAndView;

import lombok.extern.log4j.Log4j2;

@Log4j2
@Controller
@RequestMapping("/")
public class HomeController {

    @RequestMapping(method = RequestMethod.GET)
    public String index(ModelMap model) {
        model.addAttribute("message", "Hello World");
        return "index";
    }

    @RequestMapping(value = "/page2", method = RequestMethod.GET)
    public ModelAndView page2(@RequestParam(value = "name", required = false, defaultValue = "World") String name, HttpServletRequest request, Principal principal) {
        ModelAndView model = new ModelAndView("page2");
        model.addObject("message", "Hello there " + name);
        model.addObject("user", principal.getName());
        model.addObject("isAdmin", request.isUserInRole("ADMIN"));
        model.addObject("isUser", request.isUserInRole("USER"));
        log.info("user: ", principal);
        log.info("role: ", request.isUserInRole("ADMIN"));
        log.info("role: ", request.isUserInRole("USER"));
        return model;
    }
}
