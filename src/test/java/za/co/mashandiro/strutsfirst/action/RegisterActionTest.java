package za.co.mashandiro.strutsfirst.action;

import org.apache.struts2.StrutsSpringTestCase;
import org.junit.Test;

import com.opensymphony.xwork2.ActionProxy;
import com.opensymphony.xwork2.ActionSupport;

public class RegisterActionTest extends StrutsSpringTestCase{
    

    @Test
    public void testExcecuteValidationPasses() throws Exception {
        //Arrange
        request.setParameter("personBean.firstName", "John");
        request.setParameter("personBean.lastName", "Doe");
        request.setParameter("personBean.email", "doej@example.com");
        request.setParameter("personBean.age", "20");

        ActionProxy registerProxy = getActionProxy("/register.action");
        RegisterAction action = (RegisterAction) registerProxy.getAction();

        assertNotNull("No ways this cant be null...", action);

        //Act

        String result = registerProxy.execute();

        //Assert
        assertEquals(ActionSupport.SUCCESS, result);
    }

    @Test
    public void testExcecuteValidationFailsMissingFirstName() throws Exception {
        //Arrange
        //request.setParameter("personBean.firstName", "John");
        request.setParameter("personBean.lastName", "Doe");
        request.setParameter("personBean.email", "doej@example.com");
        request.setParameter("personBean.age", "20");

        ActionProxy registerProxy = getActionProxy("/register.action");
        RegisterAction action = (RegisterAction) registerProxy.getAction();

        assertNotNull("No ways this cant be null...", action);

        //Act

        String result = registerProxy.execute();

        //Assert
        assertEquals(ActionSupport.INPUT, result);
    }

}
