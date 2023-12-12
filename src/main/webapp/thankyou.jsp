<%@ taglib prefix="s" uri="/struts-tags" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Registration Successful</title>
    </head>
    <body>
        <h3><s:text name="thankyou" /></h3>
        <p>Your registration information: </p>
        <p><s:property value="personBean"/></p>
        <p><s:a href="index">Home</s:a></p>
        <div>
            <s:if test="personBean.over21">
                <s:text name="over21" />
            </s:if>
            <s:else>
                <s:text name="under21" />
            </s:else>
        </div>

    <hr/>
    <s:text name="contact" />
    </body>
</html>