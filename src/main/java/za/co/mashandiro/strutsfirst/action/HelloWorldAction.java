package za.co.mashandiro.strutsfirst.action;

import java.util.Map;

import org.apache.struts2.interceptor.SessionAware;

import com.opensymphony.xwork2.ActionSupport;

import za.co.mashandiro.strutsfirst.model.MessageStore;

public class HelloWorldAction extends ActionSupport implements SessionAware{
    private MessageStore messageStore;
    private String userName;
    private Map<String, Object> userSession;

    public String execute() throws Exception{
        messageStore = new MessageStore();
        messageStore.setMessage(messageStore.getMessage()+ " " + userName);
        System.out.println("just testing: "+ messageStore.getMessage());
        
        increaseHelloCount();

        return SUCCESS;
    }

    private void increaseHelloCount() {
        Integer helloCount = (Integer) userSession.get("helloCount");
        if(helloCount == null) {
            helloCount = 1;
        }else {
            helloCount++;
        }
        userSession.put("helloCount", helloCount);
    }

    public MessageStore getMessageStore() {
        return messageStore;
    }

    public String getUserName() {
        return userName;
    }

    public void setUserName(String userName) {
        this.userName = userName;
    }

    @Override
    public void setSession(Map<String, Object> session) {
        userSession = session;
    }
}
