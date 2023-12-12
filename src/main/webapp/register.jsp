<%@ taglib prefix="s" uri="/struts-tags" %>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Register</title>
        <s:head />
    </head>
    <body>
        <h3>Register for some prize and win many things.</h3>
        <s:form action="register" theme="css_xhtml">
            <s:textfield key="personBean.firstName" />
            <s:textfield key="personBean.lastName" />
            <s:textfield key="personBean.email" />
            <s:textfield key="personBean.age" />
            <s:select key="personBean.favoriteSport" list="sports" />
            <s:radio key="personBean.gender" list="genders" />
            <s:checkbox key="personBean.over21" />
            <s:submit />
        </s:form>

        <div>
            <h4> To test iterator things...</h4>
            <table>
                <s:iterator value="sports">
                    <tr>
                        <td><s:property /></td>
                    </tr>
                </s:iterator>
            </table>
        </div>

        <hr/>
        <s:text name="contact" />
        <s:debug/>
    </body>
</html>