package za.co.mashandiro.strutsfirst.action;

import com.opensymphony.xwork2.ActionSupport;

import za.co.mashandiro.strutsfirst.model.MessageStore;

public class HelloWorldAction extends ActionSupport{
    private MessageStore messageStore;
    private static int helloCount = 0;
    private String userName;

    public String execute() throws Exception{
        messageStore = new MessageStore();
        messageStore.setMessage(messageStore.getMessage()+ " " + userName);
        System.out.println("just testing: "+ messageStore.getMessage());
        helloCount++;
        return SUCCESS;
    }

    public MessageStore getMessageStore() {
        return messageStore;
    }

    public int getHelloCount() {
        return helloCount;
    }

    public String getUserName() {
        return userName;
    }

    public void setUserName(String userName) {
        this.userName = userName;
    }
}
